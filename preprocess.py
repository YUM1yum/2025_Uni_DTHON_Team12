# preprocessing.py

import os
import json
from glob import glob
from typing import List, Dict, Any

from torch.utils.data import Dataset, DataLoader


def find_jsons(json_dir: str) -> List[str]:
    """지정된 디렉토리에서 .json 파일 리스트를 정렬해서 반환."""
    if not os.path.isdir(json_dir):
        raise FileNotFoundError(f"json_dir not found: {json_dir}")
    return sorted(glob(os.path.join(json_dir, "*.json")))


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_visual_ann(a: dict) -> bool:
    """
    annotation 중 실제로 질의/시각요소로 사용하는 것만 필터링.
    - class_id 가 'V'로 시작하거나
    - class_name 에 표/차트 관련 문자열이 들어있는 경우
    - visual_instruction 이 비어있지 않은 경우
    """
    cid = str(a.get("class_id", "") or "")
    cname = str(a.get("class_name", "") or "")
    has_q = bool(str(a.get("visual_instruction", "") or "").strip())
    looks_visual = cid.startswith("V") or any(
        k in cname for k in ["표", "차트", "그래프", "chart", "table"]
    )
    return has_q and looks_visual


class UniDQueryDataset(Dataset):
    """
    JSON → (img_path, query_text, bbox, query_id, orig_size) 로 바꿔주는 Dataset.
    이미지는 여기서 열지 않고, path만 넘겨서 train/test 쪽에서 필요할 때 연다.
    """

    def __init__(self, json_dir: str, jpg_root: str):
        super().__init__()
        self.samples: List[Dict[str, Any]] = []
        self._build_index(json_dir, jpg_root)

    def _build_index(self, json_dir: str, jpg_root: str):
        json_files = find_jsons(json_dir)

        for jf in json_files:
            data = read_json(jf)
            src = data.get("source_data_info", {})
            jpg_name = src.get("source_data_name_jpg", None)
            if not jpg_name:
                continue
            img_path = os.path.join(jpg_root, jpg_name)
            if not os.path.exists(img_path):
                # 필요하다면 여기서 더 복잡한 매핑 로직 추가 가능
                continue

            # 문서 해상도
            doc_res = src.get("document_resolution", None)
            if isinstance(doc_res, list) and len(doc_res) == 2:
                W, H = int(doc_res[0]), int(doc_res[1])
            else:
                # fallback: 실제 이미지 열어서 사이즈 확인할 수도 있지만,
                # 여기서는 doc_res가 있다고 가정
                W = H = None

            ann_list = data.get("learning_data_info", {}).get("annotation", [])
            for a in ann_list:
                if not is_visual_ann(a):
                    continue
                bbox = a.get("bounding_box", None)
                qtxt = str(a.get("visual_instruction", "")).strip()
                qid = a.get("instance_id", "")
                self.samples.append(
                    {
                        "img_path": img_path,
                        "query_text": qtxt,
                        "bbox": bbox,  # [x, y, w, h] or None
                        "query_id": qid,
                        "orig_size": (W, H),
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def build_dataloader(
    json_dir: str,
    jpg_root: str,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
):
    """
    train.py / test.py 에서 공통으로 쓰는 DataLoader 헬퍼.
    collate_fn 을 identity 로 둬서 batch 가 "샘플 dict 리스트" 그대로 올라오도록 한다.
    """
    ds = UniDQueryDataset(json_dir=json_dir, jpg_root=jpg_root)

    def _identity_collate(batch):
        # batch: List[Dict], 그대로 넘긴다
        return batch

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_identity_collate,
    )
    return ds, dl
