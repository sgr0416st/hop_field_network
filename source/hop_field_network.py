import numpy as np
import cv2


class HopFieldNetwork:
    """
    the class to calculate HopFieldNetwork.
    """

    def __init__(self, size, theta=0.0):
        """construct HopFieldNetwork for network size, neuron's threshold.

        Args:
            size(int): network size or image size.
            theta(float): each neuron's threshold. default theta = 0.
        """
        self.size=size
        self.theta = theta
        self.W = np.zeros((size, size), dtype='int8')
        self.counter = 0
        self.w_size = 200

        self.img_size = (self.w_size, self.w_size)

    def train(self, data_list):
        """train network using data_list.

        network remember data_list patterns.

        Args:
            data_list(ndarray): data list you want to remember.

        Returns:
            None.

        """
        for data in data_list:
            data = np.reshape(data, (1, self.size))
            self.W +=  data.T.dot(data)
        for i in range(len(self.W)):
            self.W[i][i] = 0
        return self.W / len(data_list)

    def recognize(self, data, loop, show_process=True):
        """ try to remember one of trained patterns form data.

        update cells in the ordered number.

        Args:
            data(ndarray): input data.
            loop(int): the number of update a neuron of network.
            show_process(bool):if True, this function remember all process.
            else, this function only remember current　state.

        Returns:
            output(ndarray): the result of recognize.
            img_list(list): if show_process=True, return all of the process data.else, return empty list.

        """
        self.counter = 0
        img_list = []
        output = data.copy()

        for _ in range(loop):
            i = self._select_node()
            y_i = -self.theta
            for j in range(self.size):
                y_i += self.W[i][j]*data[j]
            if y_i < 0:
                output[i] = -1
            elif y_i > 0:
                output[i] = 1

            if show_process:
                print(self.counter, ": ", output)
                img = output.copy().tolist()
                img_list.append(img)

        return output, img_list

    def print_weights(self):
        """print network weight variables.

        """
        print("W: \n", self.W)

    # 今回は1から順番に活性化（更新）させる
    def _select_node(self):
        self.counter += 1
        return (self.counter-1)%self.size


# opencvで結果を描画するための関数。
def show_image_for_opencv(img_list, img_col, img_row):
    """show process image using openCV.

    the image integrated start image, recognize process image and last image.
    be careful that you have to satisfy the condition
        img_col * img_row >=  len(img_list)

    Args:
        img_list(list): list of recognize process data.
        img_col(int): the number integrated one column.
        img_row(int): the number integrated one line.

    Returns:
        None

    """
    # サイズの指定・計算
    img_len = 160
    margin = 40
    margin_gray_level = 100
    marged_width = img_len*img_col + margin*(image_col + 1)
    img_size = (img_len, img_len)
    blank_size = (img_len, margin)
    blank_size_2 = (margin, marged_width)
    ar_strat_pt = (round(margin/4), round(img_len/2))
    ar_end_pt =  (round(margin*3/4), round(img_len/2))
    # contours = np.array([[0, 0], [0, img_len], [margin, img_len], [margin, 0]])
    contours = np.array([[1, 1], [1, margin], [img_len, margin], [img_len, 1]])
    # ブランク画像の作成
    img_margin_1 = np.full(blank_size, margin_gray_level, dtype=np.uint8)
    img_margin_arrow = img_margin_1.copy()
    img_margin_arrow = cv2.arrowedLine(
        img_margin_arrow, ar_strat_pt, ar_end_pt, color=255, thickness=3, tipLength=0.3)
    img_margin_2 = np.full(blank_size_2, margin_gray_level, dtype=np.uint8)
    img_marge = np.full((0, marged_width), margin_gray_level, dtype=np.uint8)
    # 各ステップで更新された画像を結合
    for col in range(img_col):
        img_line = np.full((img_len, 0), margin_gray_level, dtype=np.uint8)
        for row in range(img_row):
            img = img_list[col*3 + row].copy()
            img_arr = None
            for i in range(len(img)):
                if img[i] == 1:
                    img[i] = 255
                else:
                    img[i] = 0
                img_arr = np.array(img).reshape(img_col, img_row).astype(np.uint8)
                img_arr = cv2.resize(img_arr, img_size, interpolation=cv2.INTER_NEAREST)
            if col == 0 and row == 0:
                img_line = cv2.hconcat([img_line, img_margin_1, img_arr])
            else:
                img_line = cv2.hconcat([img_line, img_margin_arrow, img_arr])
        img_line = cv2.hconcat([img_line, img_margin_1])
        img_marge = cv2.vconcat([img_marge, img_margin_2, img_line])
    img_marge = cv2.vconcat([img_marge, img_margin_2])

    cv2.imshow("process", img_marge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
sample program for train and recognize for using hop field network.
"""
if __name__ == '__main__':
    training_data =[[
         1, -1,  1,
        -1,  1, -1,
         1, -1,  1,
        ]]
    test_data = [
        -1,  1, -1,
        -1,  1,  1,
         1, -1,  1
    ]

    print("-------------set up-------------------")
    training_data = np.array(training_data)
    test_data = np.array(test_data)
    image_col = 3
    image_row = 3
    print("training_data: \n", training_data)
    hf = HopFieldNetwork(len(training_data[0]))
    print("------------- training step -------------------")
    hf.train(training_data)
    print("training step done.")
    print("(1)")
    hf.print_weights()
    print("------------- recognize step -------------------")
    print("(2)")
    print("input: ", test_data)
    output, img_list = hf.recognize(test_data, len(test_data))
    print("output: ", output)
    print("------------ show image ----------------")
    print("show process image for openCV. please check.")
    print("after check, please push enter on the image window.")
    show_image_for_opencv(img_list, image_col, image_row)
    print("------------ finish ----------------")
