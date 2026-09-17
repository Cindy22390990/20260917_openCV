import cv2
import json
import argparse
import sys
from pathlib import Path


def locate_part(
    input_path: Path,
    output_path: Path
):
    output_path.parent.mkdir(parents=True,exist_ok=True)

    image = cv2.imread(str(input_path))


    if image is None:raise ValueError("圖片讀取失敗")
    cv2.imshow("Original", image)
    # -----------------------------
    # 顯示圖片資訊
    # -----------------------------
    height, width, channel = image.shape
    print(f"圖片高度：{height}")
    print(f"圖片寬度：{width}")
    print(f"圖片通道：{channel}")
    image_area = height * width
    # -----------------------------
    # 灰階
    # -----------------------------
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)
    # -----------------------------
    # 高斯模糊
    # -----------------------------
    blurred = cv2.GaussianBlur(
        gray,
        (5,5),
        0
    )
    cv2.imshow("blurred", blurred)
    # -----------------------------
    # Canny
    # -----------------------------
    edges = cv2.Canny(
        blurred,
        30,
        100
    )
    cv2.imshow("edges", edges)
    # -----------------------------
    # 找輪廓
    # -----------------------------
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        raise ValueError(
            "沒有找到輪廓"
        )

    # -----------------------------
    # 篩選輪廓
    # -----------------------------
    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        # 面積太小
        if area < 1000:
            continue

        # 面積太大(超過圖片80%)
        if area > image_area * 0.8:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        ratio = w / h

        # 長寬比限制
        if ratio < 0.8 or ratio > 2.5:
            continue

        candidates.append(contour)

    if len(candidates) == 0:
        raise ValueError("沒有符合條件的輪廓")
    # -----------------------------
    # 最大候選輪廓
    # -----------------------------
    largest_contour = max(
        candidates,
        key=cv2.contourArea
    )
    area = cv2.contourArea(largest_contour)

    print(f"最大輪廓面積：{area}")
    x, y, width, height = cv2.boundingRect(
        largest_contour
    )
    # -----------------------------
    # Padding
    # -----------------------------
    padding = 10

    x = max(0, x - padding)
    y = max(0, y - padding)

    w = min(image.shape[1] - x, width + padding * 2)
    h = min(image.shape[0] - y, height + padding * 2)

    # -----------------------------
    # ROI 太小
    # -----------------------------
    if w < 50 or h < 50:
        raise ValueError("ROI 太小")

    # -----------------------------
    # ROI 太大
    # -----------------------------
    if w > image.shape[1] * 0.9:
        raise ValueError("ROI 寬度過大")

    if h > image.shape[0] * 0.9:
        raise ValueError("ROI 高度過大")

    # -----------------------------
    # 裁切 ROI
    # -----------------------------
    roi = image[
        y:y + h,
        x:x + w
    ]

    if roi.size == 0:
        raise ValueError("ROI 為空")

    cv2.imshow("ROI", roi)
    # -----------------------------
    # 儲存 ROI
    # -----------------------------

    cv2.imwrite(
            str(output_path),
            roi
        )
    # -----------------------------
    # 回傳 JSON
    # -----------------------------
    result = {

        "status": "success",

        "output": str(output_path),

        "x": x,

        "y": y,

        "width": width,

        "height": height
    }


    return result



def main():

    parser = argparse.ArgumentParser(
        description="找出圖片中最大的輪廓並裁切 ROI"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="輸入圖片路徑"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="ROI 輸出圖片路徑"
    )

    args = parser.parse_args()
    try:
        result = locate_part(
            Path(args.input),
            Path(args.output)
        )


        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=4
            )
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 0
    except Exception as e:
        print(
            json.dumps(
                {
                    "status": "error",
                    "message": str(e)
                },
                ensure_ascii=False,
                indent=4
            )
        )
        cv2.destroyAllWindows()
        return 1
if __name__ == "__main__":
    sys.exit(main())