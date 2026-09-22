import cv2
import json
import argparse
import sys
from pathlib import Path


class PartLocator:

    def __init__(
        self,
        padding=10,
        min_area=1000,
        max_area_ratio=0.8,
        min_ratio=0.8,
        max_ratio=2.5
    ):

        self.padding = padding
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

        self.image = None


    # -----------------------------
    # 讀取圖片
    # -----------------------------
    def load_image(self, input_path: Path):

        self.image = cv2.imread(
            str(input_path)
        )

        if self.image is None:
            raise ValueError(
                "圖片讀取失敗"
            )


        height, width, channel = self.image.shape

        print(f"圖片高度：{height}")
        print(f"圖片寬度：{width}")
        print(f"圖片通道：{channel}")


        cv2.imshow(
            "Original",
            self.image
        )


    # -----------------------------
    # 前處理
    # -----------------------------
    def preprocess(self):

        gray = cv2.cvtColor(
            self.image,
            cv2.COLOR_BGR2GRAY
        )


        cv2.imshow(
            "gray",
            gray
        )


        blurred = cv2.GaussianBlur(
            gray,
            (5,5),
            0
        )


        edges = cv2.Canny(
            blurred,
            30,
            100
        )


        cv2.imshow(
            "edges",
            edges
        )


        return edges



    # -----------------------------
    # 找候選輪廓
    # -----------------------------
    def find_candidates(
        self,
        edges
    ):

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        if len(contours) == 0:

            raise ValueError(
                "沒有找到輪廓"
            )


        image_area = (
            self.image.shape[0]
            *
            self.image.shape[1]
        )


        candidates = []


        for contour in contours:


            area = cv2.contourArea(
                contour
            )


            # 面積太小
            if area < self.min_area:
                continue


            # 面積太大
            if area > image_area * self.max_area_ratio:
                continue



            x,y,w,h = cv2.boundingRect(
                contour
            )


            ratio = w / h


            if (
                ratio < self.min_ratio
                or
                ratio > self.max_ratio
            ):
                continue


            candidates.append(
                contour
            )


        if not candidates:

            raise ValueError(
                "沒有符合條件的輪廓"
            )


        return candidates



    # -----------------------------
    # 找最大輪廓
    # -----------------------------
    def select_best_contour(
        self,
        candidates
    ):


        contour = max(
            candidates,
            key=cv2.contourArea
        )


        area = cv2.contourArea(
            contour
        )


        print(
            f"最大輪廓面積:{area}"
        )


        return contour



    # -----------------------------
    # ROI裁切
    # -----------------------------
    def crop_roi(
        self,
        contour
    ):


        x,y,w,h = cv2.boundingRect(
            contour
        )


        x = max(
            0,
            x-self.padding
        )

        y = max(
            0,
            y-self.padding
        )


        w = min(
            self.image.shape[1]-x,
            w+self.padding*2
        )


        h = min(
            self.image.shape[0]-y,
            h+self.padding*2
        )



        if w < 50 or h < 50:

            raise ValueError(
                "ROI太小"
            )


        if (
            w > self.image.shape[1]*0.9
            or
            h > self.image.shape[0]*0.9
        ):

            raise ValueError(
                "ROI太大"
            )


        roi = self.image[
            y:y+h,
            x:x+w
        ]


        if roi.size == 0:

            raise ValueError(
                "ROI為空"
            )


        cv2.imshow(
            "ROI",
            roi
        )


        return roi,x,y,w,h



    # -----------------------------
    # 儲存結果
    # -----------------------------
    def save_result(
        self,
        roi,
        output_path
    ):

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )


        cv2.imwrite(
            str(output_path),
            roi
        )



    # -----------------------------
    # 主流程
    # -----------------------------
    def run(
        self,
        input_path,
        output_path
    ):


        self.load_image(
            input_path
        )


        edges = self.preprocess()


        candidates = self.find_candidates(
            edges
        )


        contour = self.select_best_contour(
            candidates
        )


        roi,x,y,w,h = self.crop_roi(
            contour
        )


        self.save_result(
            roi,
            output_path
        )


        return {

            "status":"success",

            "output":str(output_path),

            "x":x,

            "y":y,

            "width":w,

            "height":h

        }





def main():


    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--input",
        required=True
    )


    parser.add_argument(
        "--output",
        required=True
    )


    args = parser.parse_args()



    locator = PartLocator()



    try:

        result = locator.run(

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
                    "status":"error",
                    "message":str(e)
                },
                ensure_ascii=False,
                indent=4
            )
        )


        cv2.destroyAllWindows()

        return 1



if __name__=="__main__":

    sys.exit(
        main()
    )