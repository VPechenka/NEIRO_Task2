from PIL import Image
from torch import Tensor, tensor, float32
from random import randint

COLOR = ["RED", "YELLOW", "GREEN", "BLUE"]
NUMBER = list(map(str, list(range(0, 10))))
ADD_TYPE = ["Pass", "Plus", "Reverse"]
DARK_ADD_TYPE = ["Color", "Plus"]

TEXT_COLOR = ["\033[31m", "\033[33m", "\033[32m", "\033[34m"]


class ImageData:
    """An image processing class."""

    def __init__(self, directory: str, file_name: str) -> None:
        self.file_name = f"{directory}/{file_name}"

        self.img = Image.new("RGB", (45, 45), (0, 0, 0))
        with Image.open(self.file_name) as img:
            # self.img = img.resize((125, 195))
            self.img.paste(img.resize((25, 39)), (10, 3))

        self.training_data = self.get_training_data()
        self.testing_data = self.get_testing_data()

        self.result: Tensor = self.__set_result__()

    def __set_result__(self) -> Tensor:
        file_name = self.file_name.split("/")[-1].split(".")[0]
        result = [1.0] * 54

        self.__get_color__(result, file_name[0])
        self.__get_number__(result, file_name[1])
        self.__get_add_type__(result, file_name[1::])

        return tensor(result, dtype=float32)

    @staticmethod
    def __get_color__(result: list[float], card_color: str):
        for index in range(len(COLOR)):
            if not COLOR[index][0] == card_color:
                for i in range(13 * index, 13 * (index + 1)):
                    result[i] = 0.0

        if not card_color == "D":
            for i in range(52, 54):
                result[i] = 0.0

    @staticmethod
    def __get_number__(result: list[float], number: str) -> None:
        for index in range(len(NUMBER)):
            if number != NUMBER[index]:
                for i in range(index, index + 39 + 1, 13):
                    result[i] = 0.0

    @staticmethod
    def __get_add_type__(result: list[float], type_str: str) -> None:
        for index in range(len(ADD_TYPE)):
            if not type_str == ADD_TYPE[index]:
                for i in range(index + 10, index + 10 + 39 + 1, 13):
                    result[i] = 0.0

        for index in range(len(DARK_ADD_TYPE)):
            if not type_str == DARK_ADD_TYPE[index]:
                result[index + 52] = 0.0

    def get_training_data(self) -> list:
        data = list()

        for degrees in range(8):
            img = self.img.copy()
            img.rotate(degrees * 45)

            data.append(
                tensor(
                    img.getdata(),
                    dtype=float32
                ).view(3, 45, 45) / 255
            )

        return data

    def get_testing_data(self) -> list:
        data = list()

        for degrees in range(8):
            img = self.img.copy()
            img.rotate(degrees * 45)

            for y in range(img.height):
                for x in range(img.width):
                    value = img.getpixel((x, y))
                    image_shared = (
                        value[0] + randint(0, 100),
                        value[1] + randint(0, 100),
                        value[2] + randint(0, 100)
                    )
                    img.putpixel((x, y), image_shared)

            data.append(
                tensor(
                    img.getdata(),
                    dtype=float32
                ).view(3, 45, 45) / 255
            )

        return data

    def get_testing_index(self):
        return self.result.argmax().item() % 8

    def result_equals(self, result: Tensor) -> int:
        if result.argmax().item() == self.result.argmax().item():
            return 1
        return 0

    def print_result(self, result: Tensor) -> None:

        result += 0 - result[result.argmin()].item()
        if result[result.argmax()].item() != 0:
            result /= result[result.argmax()].item()

        if self.result_equals(result):
            print(f"\n\033[32mPASS\033[0m ({self.file_name})")
        else:
            print(f"\n\033[31mFAILED\033[0m ({self.file_name})")

        max_indexes = result.argsort().__reversed__()[:5]
        for index in max_indexes:
            print(
                str(round(result[index].item(), 2)).rjust(4), end=" - "
            )
            self.print_index_result(index)

    @staticmethod
    def print_index_result(index: int):
        color_index = index // 13
        result = "["
        if color_index < len(COLOR):
            result += TEXT_COLOR[color_index]
            result += COLOR[color_index].ljust(6)
            result += ", "

            type_index = index % 13
            if type_index < 10:
                result += NUMBER[type_index].ljust(7)
            else:
                result += ADD_TYPE[type_index - 10].ljust(7)
        else:
            result += "BLACK".ljust(6)
            result += ", "
            result += DARK_ADD_TYPE[index - 52].ljust(7)

        result += "\033[0m]"
        print(result)


if __name__ == "__main__":
    ImageData("materials/original_set", "DColor.jpg").testing_data(4)
