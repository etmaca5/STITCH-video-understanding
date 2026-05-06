"""Temporal grounding and QA evaluation metrics."""

import re
from functools import lru_cache

import numpy as np


def temporal_iou(pred_start, pred_end, gt_start, gt_end):
    """Intersection-over-union between two temporal segments."""
    intersection = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _best_iou(pred_start, pred_end, gt_windows):
    """Max IoU of a single prediction against all ground-truth windows."""
    return max(
        temporal_iou(pred_start, pred_end, gs, ge) for gs, ge in gt_windows
    )


def compute_recall_at_k(predictions_per_query, gt_windows_per_query,
                        k, iou_threshold):
    """Recall@K at a given IoU threshold.

    Args:
        predictions_per_query: list of lists.  Each inner list contains
            prediction dicts ``{"start", "end", "score"}`` sorted by
            score descending.
        gt_windows_per_query: list of lists of ``(start, end)`` tuples.
        k: number of top predictions to consider.
        iou_threshold: minimum IoU to count as a hit.

    Returns:
        Recall value in [0, 1].
    """
    assert len(predictions_per_query) == len(gt_windows_per_query)

    assert len(predictions_per_query) > 0, "No queries to evaluate"

    hits = 0
    for preds, gts in zip(predictions_per_query, gt_windows_per_query):
        assert gts, "Every query must have at least one ground-truth window"
        top_k = preds[:k]
        for p in top_k:
            if _best_iou(p["start"], p["end"], gts) >= iou_threshold:
                hits += 1
                break

    return hits / len(gt_windows_per_query)


def _average_precision(predictions, gt_windows, iou_threshold):
    """AP for a single query (predictions already sorted by score desc)."""
    assert gt_windows, "Every query must have at least one ground-truth window"

    gt_matched = [False] * len(gt_windows)
    tp = []
    for p in predictions:
        best_idx = -1
        best_iou = 0.0
        for i, (gs, ge) in enumerate(gt_windows):
            iou = temporal_iou(p["start"], p["end"], gs, ge)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_iou >= iou_threshold and best_idx >= 0 and not gt_matched[best_idx]:
            gt_matched[best_idx] = True
            tp.append(1)
        else:
            tp.append(0)

    tp = np.array(tp, dtype=float)
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1 - tp)
    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / len(gt_windows)

    # Prepend (recall=0, precision=1) for the integral.
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])

    # Use all-points interpolation (monotonically decreasing precision).
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Area under the precision-recall curve.
    indices = np.where(np.diff(recall))[0] + 1
    ap = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])
    return float(ap)


def compute_map_at_iou(predictions_per_query, gt_windows_per_query,
                       iou_threshold):
    """Mean Average Precision at a given IoU threshold.

    Args:
        predictions_per_query: list of lists of prediction dicts
            ``{"start", "end", "score"}``, sorted by score descending.
        gt_windows_per_query: list of lists of ``(start, end)`` tuples.
        iou_threshold: minimum IoU to count a prediction as a true positive.

    Returns:
        mAP value in [0, 1].
    """
    assert len(predictions_per_query) == len(gt_windows_per_query)

    aps = []
    for preds, gts in zip(predictions_per_query, gt_windows_per_query):
        aps.append(_average_precision(preds, gts, iou_threshold))

    assert aps, "No queries to evaluate"
    return float(np.mean(aps))


def compute_mean_iou(predictions_per_query, gt_windows_per_query):
    """Standard mean IoU using the top-1 prediction per query.

    For each query, compute the best IoU of the highest-scoring prediction
    against the ground-truth windows, then average across queries.
    """
    assert len(predictions_per_query) == len(gt_windows_per_query)

    per_query = []
    for preds, gts in zip(predictions_per_query, gt_windows_per_query):
        assert gts, "Every query must have at least one ground-truth window"
        if not preds:
            per_query.append(0.0)
            continue
        top_pred = preds[0]
        per_query.append(_best_iou(top_pred["start"], top_pred["end"], gts))

    assert per_query, "No queries to evaluate"
    return float(np.mean(per_query))


def _binary_average_precision(y_true, y_score):
    """Average precision for binary labels (all-points interpolation)."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_score = np.asarray(y_score, dtype=np.float32)
    assert y_true.ndim == 1 and y_score.ndim == 1
    assert len(y_true) == len(y_score)

    positives = int(y_true.sum())
    if positives == 0:
        return 0.0

    order = np.argsort(-y_score)
    y_true = y_true[order]

    tp = np.cumsum(y_true)
    fp = np.cumsum(1.0 - y_true)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / positives

    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    indices = np.where(np.diff(recall))[0] + 1
    ap = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])
    return float(ap)


def _windows_to_clip_scores(predictions, duration, clip_length=2.0):
    """Project temporal window predictions to per-clip saliency scores."""
    num_clips = max(1, int(duration / clip_length))
    clip_scores = np.zeros(num_clips, dtype=np.float32)

    for pred in predictions:
        start = float(pred["start"])
        end = float(pred["end"])
        score = float(pred["score"])
        if end <= start:
            continue

        start_idx = max(0, int(np.floor(start / clip_length)))
        end_idx = min(num_clips - 1, int(np.floor((end - 1e-9) / clip_length)))
        if end_idx < start_idx:
            continue

        for idx in range(start_idx, end_idx + 1):
            c_start = idx * clip_length
            c_end = c_start + clip_length
            overlap = max(0.0, min(end, c_end) - max(start, c_start))
            if overlap > 0:
                weighted = score * (overlap / clip_length)
                clip_scores[idx] = max(clip_scores[idx], weighted)

    cmin = float(clip_scores.min())
    cmax = float(clip_scores.max())
    if cmax > cmin:
        clip_scores = (clip_scores - cmin) / (cmax - cmin)
    return clip_scores


def compute_qvhighlight_highlight_metrics(
    predictions_per_query,
    metadata_per_query,
    clip_length=2.0,
    min_score_thresholds=(2, 3, 4),
):
    """Compute HL-Hit1 and HL-mAP for QVHighlights-style saliency labels."""
    assert len(predictions_per_query) == len(metadata_per_query)

    out = {}
    for min_score in min_score_thresholds:
        hit_scores = []
        ap_scores = []

        for preds, meta in zip(predictions_per_query, metadata_per_query):
            rel_ids = np.asarray(meta.get("relevant_clip_ids", []), dtype=int)
            sal = np.asarray(meta.get("saliency_scores", []), dtype=np.float32)
            duration = float(meta.get("duration", 0.0))
            if duration <= 0 or rel_ids.size == 0 or sal.size == 0:
                continue

            if sal.ndim == 1:
                sal = sal[:, None]
            num_workers = sal.shape[1]
            num_clips = max(1, int(duration / clip_length))
            gt_full = np.zeros((num_clips, num_workers), dtype=np.float32)

            valid = (rel_ids >= 0) & (rel_ids < num_clips)
            rel_ids = rel_ids[valid]
            sal = sal[valid]
            if rel_ids.size == 0:
                continue
            gt_full[rel_ids] = sal

            pred_scores = _windows_to_clip_scores(
                preds, duration=duration, clip_length=clip_length
            )
            pred_top = int(np.argmax(pred_scores))
            gt_bin = (gt_full >= float(min_score)).astype(np.float32)

            hit_scores.append(float(np.max(gt_bin[pred_top])))
            for w in range(num_workers):
                ap_scores.append(
                    _binary_average_precision(gt_bin[:, w], pred_scores)
                )

        if hit_scores:
            out[f"HL-Hit1_min{min_score}"] = float(np.mean(hit_scores))
        if ap_scores:
            out[f"HL-mAP_min{min_score}"] = float(np.mean(ap_scores))

    return out


# ---------------------------------------------------------------------------
# Video QA metrics
# ---------------------------------------------------------------------------

ACTIVITYNET_QA_TYPE_NAMES = {
    0: "Motion",
    1: "Spatial",
    2: "Temporal",
    3: "Yes/No",
    4: "Color",
    5: "Object",
    6: "Location",
    7: "Number",
    8: "Other",
}


def exact_match_accuracy(predicted_answer: str, gt_answer: str) -> float:
    """Return 1.0 only when the answers are identical."""
    return float(str(predicted_answer) == str(gt_answer))


def extract_mcq_letter(text: str, valid_letters: str = "ABCD") -> str | None:
    """Extract a multiple-choice answer letter from free-form model text."""
    letters = "".join(re.escape(letter) for letter in str(valid_letters).upper())
    matches = re.findall(rf"\b([{letters}])\b", str(text).upper())
    if matches:
        return matches[0]
    return None


def mcq_letter_match(
    predicted_answer: str,
    gt_answer: str,
    valid_letters: str = "ABCD",
) -> float:
    """Return 1.0 when the parsed MCQ letter matches the gold letter."""
    pred = extract_mcq_letter(predicted_answer, valid_letters=valid_letters)
    gt = extract_mcq_letter(gt_answer, valid_letters=valid_letters)
    if pred is None or gt is None:
        return 0.0
    return float(pred == gt)


def _slug_metric_suffix(value: str) -> str:
    """Normalize free-form labels for metric names."""
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return text.strip("_") or "unknown"


def compute_videomme_mcq_metrics(per_query_results: list[dict]) -> dict:
    """Compute Video-MME multiple-choice accuracy metrics."""
    assert per_query_results, "No queries to evaluate"

    total_correct = 0.0
    duration_counts: dict[str, int] = {}
    duration_correct: dict[str, float] = {}
    domain_counts: dict[str, int] = {}
    domain_correct: dict[str, float] = {}
    task_counts: dict[str, int] = {}
    task_correct: dict[str, float] = {}

    for qr in per_query_results:
        pred = qr["predicted_answer"]
        gt = qr["gt_answer"]
        meta = qr.get("metadata", {})
        options = list(meta.get("options") or [])
        valid_letters = (
            "".join(chr(ord("A") + idx) for idx in range(len(options)))
            if options
            else "ABCD"
        )
        correct = mcq_letter_match(pred, gt, valid_letters=valid_letters)
        total_correct += correct

        duration_label = str(meta.get("duration_label", "")).strip().lower()
        if duration_label:
            duration_counts[duration_label] = duration_counts.get(duration_label, 0) + 1
            duration_correct[duration_label] = (
                duration_correct.get(duration_label, 0.0) + correct
            )

        domain = str(meta.get("domain", "")).strip()
        if domain:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            domain_correct[domain] = domain_correct.get(domain, 0.0) + correct

        task_type = str(meta.get("task_type", "")).strip()
        if task_type:
            task_counts[task_type] = task_counts.get(task_type, 0) + 1
            task_correct[task_type] = task_correct.get(task_type, 0.0) + correct

    total_count = len(per_query_results)
    metrics = {
        "Accuracy": total_correct / total_count,
    }

    for duration_label in ("short", "medium", "long"):
        count = duration_counts.get(duration_label, 0)
        if count > 0:
            metrics[f"Accuracy_{duration_label}"] = duration_correct[duration_label] / count

    for domain, count in sorted(domain_counts.items()):
        metrics[f"Accuracy_domain_{_slug_metric_suffix(domain)}"] = (
            domain_correct[domain] / count
        )

    for task_type, count in sorted(task_counts.items()):
        metrics[f"Accuracy_task_{_slug_metric_suffix(task_type)}"] = (
            task_correct[task_type] / count
        )

    return metrics


# Official LVBench category abbreviations (from scripts/test_acc.py).
_LVBENCH_CATEGORY_ABBREV = {
    "key information retrieval": "KIR",
    "event understanding": "EU",
    "summarization": "Sum",
    "entity recognition": "ER",
    "reasoning": "Rea",
    "temporal grounding": "TG",
}


def compute_lvbench_metrics(per_query_results: list[dict]) -> dict:
    """Compute LVBench MCQ accuracy, replicating the official metric.

    Overall accuracy counts each QA once.  Per-category accuracy counts each
    QA once *per tag* in its ``question_type`` list (a multi-tagged question
    increments every relevant category bucket independently).
    """
    assert per_query_results, "No queries to evaluate"

    total_correct = 0.0
    category_correct: dict[str, float] = {}
    category_total: dict[str, int] = {}

    for qr in per_query_results:
        pred_letter = qr.get("predicted_answer_letter")
        gt = str(qr["gt_answer"]).strip().upper()

        if pred_letter is None:
            correct = 0.0
        else:
            correct = float(str(pred_letter).strip().upper() == gt)
        total_correct += correct

        question_types = list(qr.get("metadata", {}).get("question_type", []))
        for cat in question_types:
            category_total[cat] = category_total.get(cat, 0) + 1
            category_correct[cat] = category_correct.get(cat, 0.0) + correct

    total_count = len(per_query_results)
    metrics: dict[str, float] = {
        "Accuracy": total_correct / total_count,
    }

    for cat in sorted(category_total.keys()):
        abbrev = _LVBENCH_CATEGORY_ABBREV.get(cat, _slug_metric_suffix(cat))
        metrics[f"Accuracy_{abbrev}"] = category_correct[cat] / category_total[cat]

    return metrics


def compute_mlvu_metrics(per_query_results: list[dict]) -> dict:
    """Compute MLVU MCQ accuracy: per-task accuracy + macro-average (M-Avg).

    Mirrors ``evaluation/multiple_choice_evaluation/choice_bench.py`` from the
    official MLVU repo (JUNJIE99/MLVU): each ``question_type`` contributes one
    accuracy number, and the headline ``Accuracy`` (M-Avg) is the unweighted
    mean across question_types — not a per-question micro-average. The
    micro-average is reported separately as ``Accuracy_micro``.
    """
    assert per_query_results, "No queries to evaluate"

    task_correct: dict[str, float] = {}
    task_total: dict[str, int] = {}

    for qr in per_query_results:
        meta = qr.get("metadata", {})
        options = list(meta.get("options") or [])
        valid_letters = (
            "".join(chr(ord("A") + i) for i in range(len(options)))
            if options
            else "ABCD"
        )
        correct = mcq_letter_match(
            qr["predicted_answer"], qr["gt_answer"], valid_letters=valid_letters,
        )
        qtype = str(meta.get("question_type", "")).strip() or "unknown"
        task_total[qtype] = task_total.get(qtype, 0) + 1
        task_correct[qtype] = task_correct.get(qtype, 0.0) + correct

    per_task = {q: task_correct[q] / task_total[q] for q in task_total}

    metrics: dict[str, float] = {
        "Accuracy": sum(per_task.values()) / len(per_task),
        "Accuracy_micro": sum(task_correct.values()) / sum(task_total.values()),
    }
    for qtype in sorted(per_task):
        metrics[f"Accuracy_{_slug_metric_suffix(qtype)}"] = per_task[qtype]
    return metrics


def parse_longvideobench_mcq_answer(
    response: str,
    valid_letters: list[str] | tuple[str, ...],
) -> str | None:
    """Parse a LongVideoBench-style MCQ response into one option letter.

    Returns ``None`` when no unambiguous letter is found. Some public
    LongVideoBench scripts fall back to a *random* choice in that case, which
    makes metrics non-reproducible and can bias accuracy; here unparseable
    outputs are treated as wrong downstream (see :func:`compute_longvideobench_metrics`).
    """
    all_choices = [str(letter).upper() for letter in valid_letters]
    if not all_choices:
        raise ValueError("valid_letters must not be empty")

    text = str(response).strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        text = text.replace(answer_prefix, "")

    letters_pattern = "".join(re.escape(letter) for letter in all_choices)
    if len(text.split()) > 10 and not re.search(rf"[{letters_pattern}]", text.upper()):
        return None

    match = re.search(rf"[{letters_pattern}]", text.upper())
    if match is None:
        return None
    return match[0]


def compute_longvideobench_metrics(per_query_results: list[dict]) -> dict:
    """Compute LongVideoBench MCQ accuracy on samples that have ``gt_answer``."""

    def _has_gt_answer(qr: dict) -> bool:
        gt = qr.get("gt_answer")
        if gt is None:
            return False
        return str(gt).strip() != ""

    labeled = [qr for qr in per_query_results if _has_gt_answer(qr)]
    assert labeled, (
        "No queries with ground-truth labels to evaluate (gt_answer missing on "
        "all samples; use val split, or only aggregate metrics on labeled rows)."
    )

    total_correct = 0.0
    duration_counts: dict[str, int] = {}
    duration_correct: dict[str, float] = {}
    category_counts: dict[str, int] = {}
    category_correct: dict[str, float] = {}

    for qr in labeled:
        metadata = qr.get("metadata", {})
        options = list(metadata.get("options") or [])
        valid_letters = tuple(chr(ord("A") + idx) for idx in range(len(options)))
        parsed_pred = qr.get("predicted_answer_letter")
        if not parsed_pred:
            if valid_letters:
                parsed_pred = parse_longvideobench_mcq_answer(
                    qr.get("predicted_answer", ""),
                    valid_letters=valid_letters,
                )
            else:
                parsed_pred = None
        gt_answer = str(qr["gt_answer"]).strip().upper()
        if parsed_pred is None:
            correct = 0.0
        else:
            correct = float(str(parsed_pred).strip().upper() == gt_answer)
        total_correct += correct

        duration_group = str(metadata.get("duration_group", "")).strip()
        if duration_group:
            duration_counts[duration_group] = duration_counts.get(duration_group, 0) + 1
            duration_correct[duration_group] = (
                duration_correct.get(duration_group, 0.0) + correct
            )

        question_category = str(metadata.get("question_category", "")).strip()
        if question_category:
            category_counts[question_category] = (
                category_counts.get(question_category, 0) + 1
            )
            category_correct[question_category] = (
                category_correct.get(question_category, 0.0) + correct
            )

    total_count = len(labeled)
    metrics = {
        "Accuracy": total_correct / total_count,
    }

    for duration_group in sorted(duration_counts.keys(), key=lambda value: int(value)):
        count = duration_counts[duration_group]
        metrics[f"Accuracy_duration_group_{duration_group}"] = (
            duration_correct[duration_group] / count
        )

    for question_category, count in sorted(category_counts.items()):
        metrics[f"Accuracy_question_category_{question_category}"] = (
            category_correct[question_category] / count
        )

    return metrics


def _compute_pass_at_k(
    similarity_matrix,
    query_target_ids: list[str],
    candidate_ids: list[str],
    topk_values: list[int],
):
    """Compute LoVR-style pass@k from a query-candidate similarity matrix."""
    if len(query_target_ids) == 0:
        return {}
    if len(candidate_ids) == 0:
        raise ValueError("candidate_ids must not be empty")

    candidate_ids = [str(value) for value in candidate_ids]
    query_target_ids = [str(value) for value in query_target_ids]
    sim = np.asarray(similarity_matrix, dtype=np.float32)
    if sim.shape != (len(query_target_ids), len(candidate_ids)):
        raise ValueError(
            "similarity_matrix shape does not match query/candidate counts: "
            f"{sim.shape} vs ({len(query_target_ids)}, {len(candidate_ids)})"
        )

    metrics = {}
    for k in topk_values:
        if k <= 0:
            raise ValueError(f"topk must be positive; got {k}")
        actual_k = min(int(k), len(candidate_ids))
        ranked = np.argsort(-sim, axis=1)[:, :actual_k]
        hits = 0
        for row_idx, target_id in enumerate(query_target_ids):
            top_ids = [candidate_ids[idx] for idx in ranked[row_idx]]
            if target_id in top_ids:
                hits += 1
        metrics[int(k)] = hits / len(query_target_ids)
    return metrics


def compute_lovr_pass_metrics(
    clip_text_to_clip_similarity,
    video_text_to_video_similarity,
    clip_video_to_text_similarity,
    video_video_to_text_similarity,
    clip_query_target_ids: list[str],
    video_query_target_ids: list[str],
    clip_candidate_ids: list[str],
    video_candidate_ids: list[str],
    topk_values: list[int],
) -> dict:
    """Compute LoVR's official pass@k metrics."""
    metrics = {}

    clip_pass = _compute_pass_at_k(
        clip_text_to_clip_similarity,
        clip_query_target_ids,
        clip_candidate_ids,
        topk_values,
    )
    full_pass = _compute_pass_at_k(
        video_text_to_video_similarity,
        video_query_target_ids,
        video_candidate_ids,
        topk_values,
    )
    v2t_clip_pass = _compute_pass_at_k(
        clip_video_to_text_similarity,
        clip_candidate_ids,
        clip_query_target_ids,
        topk_values,
    )
    v2t_full_pass = _compute_pass_at_k(
        video_video_to_text_similarity,
        video_candidate_ids,
        video_query_target_ids,
        topk_values,
    )

    for k in topk_values:
        metrics[f"clip_pass@{k}"] = clip_pass[int(k)]
        metrics[f"full_pass@{k}"] = full_pass[int(k)]
        metrics[f"v2t_clip_pass@{k}"] = v2t_clip_pass[int(k)]
        metrics[f"v2t_full_pass@{k}"] = v2t_full_pass[int(k)]

    # TODO: LoVR's public scripts can optionally report theme retrieval metrics
    # when theme-specific text features are available. This repo currently
    # implements the core four retrieval directions only.
    return metrics


# ---------------------------------------------------------------------------
# GEBD (Generic Event Boundary Detection) metrics
# ---------------------------------------------------------------------------
# Exact reimplementation of the official LOVEU Challenge evaluation code:
# https://github.com/StanLei52/GEBD/blob/main/Challenge_eval_Code/eval.py


def _gebd_f1_single_rater(
    gt_timestamps: list[float],
    det_timestamps: list[float],
    threshold: float,
    video_duration: float,
) -> tuple[int, int, int]:
    """Greedy matching of detections to one rater's GT boundaries.

    Returns (tp, num_pos, num_det).
    """
    num_pos = len(gt_timestamps)
    num_det = len(det_timestamps)
    if num_pos == 0 or num_det == 0:
        return 0, num_pos, num_det

    offset_arr = np.zeros((num_pos, num_det))
    for gi in range(num_pos):
        for di in range(num_det):
            offset_arr[gi, di] = abs(gt_timestamps[gi] - det_timestamps[di])

    tp = 0
    for gi in range(num_pos):
        if offset_arr.shape[1] == 0:
            break
        min_idx = np.argmin(offset_arr[gi, :])
        if offset_arr[gi, min_idx] <= threshold * video_duration:
            tp += 1
            offset_arr = np.delete(offset_arr, min_idx, 1)

    return tp, num_pos, num_det


def compute_gebd_f1_metrics(
    per_video_results: list[dict],
    thresholds: list[float] | None = None,
) -> dict:
    """Compute GEBD F1 metrics matching the official challenge evaluation.

    Each entry in *per_video_results* must have:
      - ``predicted_boundaries``: list of timestamps in seconds
      - ``metadata.gt_boundaries_per_rater``: list of lists of timestamps
      - ``metadata.video_duration``: video length in seconds

    Args:
        per_video_results: one dict per video.
        thresholds: Rel.Dis thresholds (default ``[0.05]``).

    Returns:
        Dict with ``F1@{t}``, ``Precision@{t}``, ``Recall@{t}`` for each
        threshold, plus ``Avg_F1`` when multiple thresholds are given.
    """
    if thresholds is None:
        thresholds = [0.05]
    assert per_video_results, "No videos to evaluate"

    metrics: dict[str, float] = {}
    f1_list = []

    for threshold in thresholds:
        tp_all = 0
        num_pos_all = 0
        num_det_all = 0

        for vr in per_video_results:
            meta = vr.get("metadata", {})
            gt_per_rater = meta.get("gt_boundaries_per_rater", [])
            det = vr.get("predicted_boundaries", [])
            duration = float(meta.get("video_duration", 0.0))

            if not gt_per_rater:
                continue

            if not det:
                num_pos_all += len(gt_per_rater[0])
                continue

            num_det = len(det)
            num_det_all += num_det

            best_f1 = -1.0
            best_tp = 0
            best_num_pos = 0

            for rater_gt in gt_per_rater:
                tp, num_pos, _ = _gebd_f1_single_rater(
                    rater_gt, det, threshold, duration,
                )
                fn = num_pos - tp
                fp = num_det - tp
                rec = 1.0 if num_pos == 0 else tp / (tp + fn)
                prec = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
                f1 = 0.0 if (rec + prec) == 0 else 2 * rec * prec / (rec + prec)

                if f1 > best_f1:
                    best_f1 = f1
                    best_tp = tp
                    best_num_pos = num_pos

            tp_all += best_tp
            num_pos_all += best_num_pos

        fn_all = num_pos_all - tp_all
        fp_all = num_det_all - tp_all
        rec = 1.0 if num_pos_all == 0 else tp_all / (tp_all + fn_all)
        prec = 0.0 if (tp_all + fp_all) == 0 else tp_all / (tp_all + fp_all)
        f1 = 0.0 if (rec + prec) == 0 else 2 * rec * prec / (rec + prec)

        metrics[f"F1@{threshold}"] = f1
        metrics[f"Precision@{threshold}"] = prec
        metrics[f"Recall@{threshold}"] = rec
        f1_list.append(f1)

    if len(thresholds) > 1:
        metrics["Avg_F1"] = float(np.mean(f1_list))

    return metrics


def _tokenize_answer(text: str) -> tuple[str, ...]:
    """Tokenize an answer into lowercase word-like tokens for WUPS."""
    return tuple(re.findall(r"[a-z0-9]+", str(text).lower()))


def _load_wordnet():
    """Import and load WordNet on demand for WUPS scoring."""
    try:
        import nltk
        from nltk.corpus import wordnet as wn
    except ImportError as exc:
        raise RuntimeError(
            "ActivityNet-QA WUPS requires the `nltk` package. "
            "Install it before running this metric."
        ) from exc

    try:
        wn.synsets("dog")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        wn.synsets("dog")
    return wn


@lru_cache(maxsize=None)
def _wordnet_synsets(word: str):
    """Cache WordNet synsets per token."""
    return tuple(_load_wordnet().synsets(word))


@lru_cache(maxsize=None)
def _max_wup_similarity(word_a: str, word_b: str) -> float:
    """Return the maximum Wu-Palmer similarity between two tokens."""
    if word_a == word_b:
        return 1.0

    synsets_a = _wordnet_synsets(word_a)
    synsets_b = _wordnet_synsets(word_b)
    if not synsets_a or not synsets_b:
        return 0.0

    best = 0.0
    for syn_a in synsets_a:
        for syn_b in synsets_b:
            sim = syn_a.wup_similarity(syn_b)
            if sim is not None and sim > best:
                best = float(sim)
    return best


def _thresholded_wup(word_a: str, word_b: str, gamma: float) -> float:
    """Apply the ActivityNet-QA paper thresholding to a word pair."""
    wup = _max_wup_similarity(word_a, word_b)
    if wup >= gamma:
        return wup
    return 0.1 * wup


def _directed_wups_score(
    source_tokens: tuple[str, ...],
    target_tokens: tuple[str, ...],
    gamma: float,
) -> float:
    """Compute one directed side of the WUPS set comparison."""
    if not source_tokens:
        return 1.0 if not target_tokens else 0.0
    if not target_tokens:
        return 0.0

    score = 1.0
    for source_word in source_tokens:
        best_match = max(
            _thresholded_wup(source_word, target_word, gamma)
            for target_word in target_tokens
        )
        score *= best_match
    return score


def compute_wups(predicted_answer: str, gt_answer: str, gamma: float) -> float:
    """Compute the paper's WUPS score for one answer pair."""
    pred_tokens = _tokenize_answer(predicted_answer)
    gt_tokens = _tokenize_answer(gt_answer)

    if pred_tokens == gt_tokens:
        return 1.0

    pred_to_gt = _directed_wups_score(pred_tokens, gt_tokens, gamma)
    gt_to_pred = _directed_wups_score(gt_tokens, pred_tokens, gamma)
    return min(pred_to_gt, gt_to_pred)


def compute_activitynet_qa_metrics(per_query_results: list[dict]) -> dict:
    """Compute the ActivityNet-QA paper metrics.

    Each entry in *per_query_results* must have ``predicted_answer``,
    ``gt_answer``, and ``answer_type`` keys.
    """
    assert per_query_results, "No queries to evaluate"

    corrects: dict[int, int] = {}
    type_count: dict[int, int] = {}
    total_wups_0 = 0.0
    total_wups_9 = 0.0

    for qr in per_query_results:
        answer_type = int(qr["answer_type"])
        type_count[answer_type] = type_count.get(answer_type, 0) + 1

        pred = qr["predicted_answer"]
        gt = qr["gt_answer"]

        if exact_match_accuracy(pred, gt):
            corrects[answer_type] = corrects.get(answer_type, 0) + 1

        total_wups_0 += compute_wups(pred, gt, gamma=0.0)
        total_wups_9 += compute_wups(pred, gt, gamma=0.9)

    metrics: dict[str, float] = {}

    total_correct = sum(corrects.values())
    total_count = sum(type_count.values())
    metrics["Accuracy"] = total_correct / total_count
    metrics["WUPS@0.0"] = total_wups_0 / total_count
    metrics["WUPS@0.9"] = total_wups_9 / total_count

    for type_id, name in ACTIVITYNET_QA_TYPE_NAMES.items():
        count = type_count.get(type_id, 0)
        if count > 0:
            metrics[f"Accuracy_{name}"] = corrects.get(type_id, 0) / count

    free_correct = sum(corrects.get(i, 0) for i in range(3, 9))
    free_total = sum(type_count.get(i, 0) for i in range(3, 9))
    if free_total > 0:
        metrics["Accuracy_Free"] = free_correct / free_total

    return metrics
