from __future__ import annotations

import cv2
import numpy as np

MAX_LONG_SIDE = 1200
TOP_PERCENT = 80  # percentile threshold for "attention zone"
MAX_ZONES = 5
MIN_ZONE_RATIO = 0.01  # ignore zones smaller than 1% of image area


def resize_if_needed(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= MAX_LONG_SIDE:
        return image
    scale = MAX_LONG_SIDE / long_side
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def compute_saliency(image: np.ndarray) -> np.ndarray:
    """SpectralResidual (40%) + FineGrained (60%) blend."""
    sr = cv2.saliency.StaticSaliencySpectralResidual_create()
    ok_sr, map_sr = sr.computeSaliency(image)

    fg = cv2.saliency.StaticSaliencyFineGrained_create()
    ok_fg, map_fg = fg.computeSaliency(image)

    if ok_sr and ok_fg:
        blended = 0.4 * map_sr + 0.6 * map_fg
    elif ok_fg:
        blended = map_fg
    elif ok_sr:
        blended = map_sr
    else:
        raise RuntimeError("サリエンシーマップの生成に失敗しました")

    blended = (blended - blended.min()) / (blended.max() - blended.min() + 1e-8)
    saliency_u8 = (blended * 255).astype(np.uint8)
    saliency_u8 = cv2.GaussianBlur(saliency_u8, (0, 0), sigmaX=15, sigmaY=15)
    return saliency_u8


def colorize_heatmap(saliency_u8: np.ndarray) -> np.ndarray:
    """Saliency grayscale -> JET colourmap (BGR)."""
    return cv2.applyColorMap(saliency_u8, cv2.COLORMAP_JET)


def detect_attention_zones(saliency_u8: np.ndarray) -> list[dict]:
    """上位 20% の注目領域を矩形で返す。"""
    threshold_val = int(np.percentile(saliency_u8, TOP_PERCENT))
    _, binary = cv2.threshold(saliency_u8, threshold_val, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = saliency_u8.shape[0] * saliency_u8.shape[1]

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    zones: list[dict] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < img_area * MIN_ZONE_RATIO:
            continue
        zones.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
        if len(zones) >= MAX_ZONES:
            break

    for i, z in enumerate(zones, 1):
        z["label"] = f"注目ゾーン{i}"

    return zones


def compute_attention_score(saliency_u8: np.ndarray) -> dict:
    """上位10%画素の面積比率から集中度を 0-100 でスコア化する。"""
    threshold_val = np.percentile(saliency_u8, 90)
    high_pixels = int(np.count_nonzero(saliency_u8 >= threshold_val))
    total_pixels = saliency_u8.shape[0] * saliency_u8.shape[1]
    area_ratio = high_pixels / total_pixels * 100  # %

    # 面積比率が小さい = 集中している = スコアが高い
    # 0% → 100, 50%+ → 0 にマッピング（線形クランプ）
    score = int(max(0, min(100, round(100 - area_ratio * 2))))

    if area_ratio <= 5:
        type_label = "集中型"
        advice = "視線が一点に集まっています。CTAや重要要素がその位置にあるか確認しましょう"
    elif area_ratio <= 15:
        type_label = "やや集中型"
        advice = "視線の集中度が高めです。メインメッセージの配置と合っているか確認を"
    elif area_ratio <= 30:
        type_label = "バランス型"
        advice = "視線がバランスよく分散しています。情報の優先順位を意識するとさらに良くなります"
    else:
        type_label = "分散型"
        advice = "視線が広く分散しています。視線を誘導するアンカーポイントを作ると改善できます"

    return {
        "score": score,
        "type": type_label,
        "area_ratio": round(area_ratio, 1),
        "advice": advice,
    }


def process_image(image_bytes: bytes) -> dict:
    """
    画像バイト列を受け取り、解析結果を dict で返す。
    - original_png: 元画像 (リサイズ後)
    - heatmap_png:  カラーヒートマップ単体 (半透明合成はフロント側)
    - zones:        注目ゾーン矩形リスト
    - score:        注目度スコア情報
    - width / height: 画像サイズ (フロント描画用)
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("画像の読み込みに失敗しました。対応形式: JPG / PNG")

    image = resize_if_needed(image)
    h, w = image.shape[:2]

    saliency_u8 = compute_saliency(image)
    heatmap_bgr = colorize_heatmap(saliency_u8)

    zones = detect_attention_zones(saliency_u8)
    score_info = compute_attention_score(saliency_u8)

    _, orig_png = cv2.imencode(".png", image)
    _, heat_png = cv2.imencode(".png", heatmap_bgr)

    return {
        "original_png": orig_png.tobytes(),
        "heatmap_png": heat_png.tobytes(),
        "zones": zones,
        "score": score_info,
        "width": w,
        "height": h,
    }
