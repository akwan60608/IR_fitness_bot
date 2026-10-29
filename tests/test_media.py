import tempfile
import unittest
from pathlib import Path

from PIL import Image

from chunking.pipeline.media import (
    media_chunk_records,
    media_records,
    picture_description_record_stats,
    picture_description_stats,
)
from chunking.pipeline.picture_enrichment import p2_picture_kind, p2_prompt


class _Provenance:
    def model_dump(self, mode: str) -> dict:
        return {"page_no": 2, "mode": mode}


class _Prediction:
    class_name = "qr_code"
    confidence = 0.98


class _Classification:
    def get_main_prediction(self):
        return _Prediction()


class _IconPrediction:
    class_name = "icon"
    confidence = 0.98


class _IconClassification:
    def get_main_prediction(self):
        return _IconPrediction()


class _FlowPrediction:
    class_name = "flow_chart"
    confidence = 0.98


class _FlowClassification:
    def get_main_prediction(self):
        return _FlowPrediction()


class _Picture:
    self_ref = "#/pictures/0"
    prov = [_Provenance()]
    captions = []
    meta = type("Meta", (), {"classification": _Classification()})()

    def get_image(self, document):
        return Image.new("RGB", (2, 2), "white")

    def caption_text(self, document):
        return ""


class _Document:
    pictures = [_Picture()]


class _Box:
    def __init__(self, area: float) -> None:
        self._area = area

    def area(self) -> float:
        return self._area


class _SizedProvenance(_Provenance):
    def __init__(self, area: float) -> None:
        self.page_no = 1
        self.bbox = _Box(area)


class _DescribedPicture(_Picture):
    self_ref = "#/pictures/1"
    prov = [_SizedProvenance(200)]
    meta = type("Meta", (), {"classification": _Classification(), "description": type("Description", (), {"text": "NO_RETRIEVAL"})()})()


class _SmallPicture(_Picture):
    self_ref = "#/pictures/2"
    prov = [_SizedProvenance(0.5)]


class _OtherPicture(_Picture):
    self_ref = "#/pictures/3"
    prov = [_SizedProvenance(200)]
    meta = type("Meta", (), {"classification": None})()


class _DescriptionDocument:
    pictures = [_Picture(), _DescribedPicture(), _SmallPicture(), _OtherPicture()]
    pages = {1: type("Page", (), {"size": type("Size", (), {"width": 100, "height": 100})()})()}

    def __init__(self) -> None:
        self.pictures[0].prov = [_SizedProvenance(200)]
        self.pictures[0].meta = type("Meta", (), {"classification": _Classification()})()


class _GeometryProvenance:
    def __init__(self, page_no: int, left: float, top: float, width: float = 10, height: float = 10) -> None:
        self.page_no = page_no
        self._bbox = {"l": left, "t": top, "r": left + width, "b": top - height}

    def model_dump(self, mode: str) -> dict:
        return {"page_no": self.page_no, "bbox": self._bbox, "mode": mode}


class _TextItem:
    def __init__(self, ref: str, text: str, label: str, provenance: _GeometryProvenance) -> None:
        self.self_ref = ref
        self.text = text
        self.label = label
        self.prov = [provenance]


class _ContextPicture(_Picture):
    self_ref = "#/pictures/9"
    prov = [_GeometryProvenance(2, 100, 100)]
    meta = type("Meta", (), {
        "classification": None,
        "description": type("Description", (), {"text": "Person holding an exercise band."})(),
    })()


class _ContextDocument:
    pictures = [_ContextPicture()]
    texts = [
        _TextItem("#/texts/0", "肩胛帶熱身", "section_header", _GeometryProvenance(2, 80, 180)),
        _TextItem("#/texts/1", "維持 30～60% 張力，快速晃動 1 分鐘。", "text", _GeometryProvenance(2, 105, 90)),
        _TextItem("#/texts/2", "其他頁面的文字。", "text", _GeometryProvenance(3, 100, 100)),
    ]


class _ActionIconPicture(_ContextPicture):
    self_ref = "#/pictures/10"
    meta = type("Meta", (), {"classification": _IconClassification()})()

    def caption_text(self, document):
        return "手心向下纏繞彈力帶後向外拉開。"


class _ActionIconDocument(_ContextDocument):
    pictures = [_ActionIconPicture()]


class _P2Describer:
    def describe(self, **_kwargs):
        return {
            "text": "人物以手心向下的握法拉開彈力帶。",
            "model": "test-vlm",
            "kind": "exercise_instruction",
            "prompt_id": "p2_exercise_instruction_v1",
            "prompt": "test prompt",
            "action_caption_override": True,
            "status": "generated",
        }


class _FlowPicture(_ContextPicture):
    self_ref = "#/pictures/11"
    prov = [_GeometryProvenance(2, 0, 100, width=100, height=100)]
    meta = type("Meta", (), {"classification": _FlowClassification()})()


class _FlowDocument:
    pictures = [_FlowPicture()]
    texts = [
        _TextItem("#/texts/0", "肌少症", "text", _GeometryProvenance(2, 2, 98)),
        _TextItem("#/texts/1", "活動量減少", "text", _GeometryProvenance(2, 50, 50)),
        _TextItem("#/texts/2", "失能", "text", _GeometryProvenance(2, 95, 5)),
        _TextItem("#/texts/3", "圖框外文字", "text", _GeometryProvenance(2, 200, 50)),
    ]


class MediaTest(unittest.TestCase):
    def test_qr_picture_is_saved_but_not_retrieval_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            records = media_records(
                _Document(),
                document_id="sha256:document",
                image_dir=data_dir / "media" / "images",
                data_dir=data_dir,
            )

        self.assertEqual(records[0]["decision"]["role"], "qr_link")
        self.assertFalse(records[0]["decision"]["retrieval_eligible"])
        self.assertIsNotNone(records[0]["image_path"])

    def test_description_becomes_a_separate_embedding_record(self) -> None:
        records = [{
            "media_id": "media-1",
            "caption_text": "圖 1：深蹲動作",
            "description": {"text": "一個人示範深蹲。"},
            "decision": {"retrieval_eligible": True},
            "source_locations": [{"page_no": 2}],
        }]

        chunks = media_chunk_records(records, document_id="sha256:document", pipeline_run_id="run-1")

        self.assertEqual(chunks[0]["media_id"], "media-1")
        self.assertIn("深蹲", chunks[0]["text"])

    def test_p2_keeps_nearby_text_with_traceable_source_references(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            records = media_records(
                _ContextDocument(),
                document_id="sha256:document",
                image_dir=data_dir / "media" / "images",
                data_dir=data_dir,
                include_context=True,
            )

        context = records[0]["context"]
        chunks = media_chunk_records(records, document_id="sha256:document", pipeline_run_id="run-p2")

        self.assertEqual(context["strategy"], "same_page_figure_context_v2")
        self.assertIn("#/texts/1", context["source_item_refs"])
        self.assertIn("30～60%", chunks[0]["text"])
        self.assertIn("#/texts/1", [location.get("doc_item_ref") for location in chunks[0]["source_locations"]])

    def test_no_retrieval_description_is_not_embedded(self) -> None:
        records = [{
            "media_id": "media-1",
            "caption_text": "裝飾圖",
            "description": {"text": " NO_RETRIEVAL "},
            "decision": {"retrieval_eligible": True},
            "source_locations": [{"page_no": 2}],
        }]

        chunks = media_chunk_records(records, document_id="sha256:document", pipeline_run_id="run-1")

        self.assertEqual(chunks, [])

    def test_picture_description_stats_reports_each_outcome(self) -> None:
        stats = picture_description_stats(_DescriptionDocument(), area_threshold=0.01)

        self.assertEqual(stats["generated"], 1)
        self.assertEqual(stats["skipped_classification"], 1)
        self.assertEqual(stats["skipped_area"], 1)
        self.assertEqual(stats["not_generated_other"], 1)

    def test_action_captioned_icon_is_described_and_retrievable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            records = media_records(
                _ActionIconDocument(),
                document_id="sha256:document",
                image_dir=data_dir / "media" / "images",
                data_dir=data_dir,
                include_context=True,
                picture_describer=_P2Describer(),
            )

        self.assertEqual(records[0]["p2"]["kind"], "exercise_instruction")
        self.assertEqual(p2_picture_kind("icon", records[0]["caption_text"]), "exercise_instruction")
        self.assertTrue(records[0]["p2"]["action_caption_override"])
        self.assertEqual(records[0]["decision"]["role"], "instructional_figure")
        self.assertTrue(records[0]["decision"]["retrieval_eligible"])
        self.assertEqual(picture_description_record_stats(records)["generated"], 1)

    def test_large_action_context_overrides_misclassified_figures_but_not_small_icons(self) -> None:
        context = "[section_header] 杠铃弓步蹲举\n[list_item] ·左脚前踏，身体慢慢下沉。"

        self.assertEqual(
            p2_picture_kind("flow_chart", None, context, area_fraction=0.1),
            "exercise_instruction",
        )
        self.assertEqual(
            p2_picture_kind("icon", None, context, area_fraction=0.001),
            "decorative",
        )
        self.assertEqual(p2_picture_kind("flow_chart", None, "[section_header] 肌少症"), "flow_chart")
        self.assertEqual(
            p2_picture_kind("qr_code", "弓步蹲举", context, area_fraction=0.1),
            "decorative",
        )

    def test_flow_chart_context_prefers_all_text_inside_its_box(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            records = media_records(
                _FlowDocument(),
                document_id="sha256:document",
                image_dir=data_dir / "media" / "images",
                data_dir=data_dir,
                include_context=True,
            )

        refs = records[0]["context"]["source_item_refs"]
        self.assertEqual(refs, ["#/texts/0", "#/texts/1", "#/texts/2"])
        self.assertNotIn("#/texts/3", refs)
        self.assertIn("A -> B", p2_prompt("flow_chart", None, "肌少症\n活動量減少"))


if __name__ == "__main__":
    unittest.main()
