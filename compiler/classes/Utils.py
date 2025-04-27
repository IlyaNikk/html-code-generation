import string
import random
import uuid

TEXT_PLACE_HOLDER = "[]"
TEXT_PLACE_HOLDER_1 = "[1]"
TEXT_PLACE_HOLDER_2 = "[2]"

class Utils:
    uuid_id = str(uuid.uuid4())

    @staticmethod
    def get_random_text(length_text=10, space_number=1, with_upper_case=True):
        results = []
        while len(results) < length_text:
            char = random.choice(string.ascii_letters[:26])
            results.append(char)
        if with_upper_case:
            results[0] = results[0].upper()

        current_spaces = []
        while len(current_spaces) < space_number:
            space_pos = random.randint(2, length_text - 3)
            if space_pos in current_spaces:
                break
            results[space_pos] = " "
            if with_upper_case:
                results[space_pos + 1] = results[space_pos - 1].upper()

            current_spaces.append(space_pos)

        return ''.join(results)

    @staticmethod
    def render_content_with_text(key, value):
        if key.find("btn") != -1:
            return value.replace(TEXT_PLACE_HOLDER, "What is Lorem?")
        elif key.find("title") != -1:
            return value.replace(TEXT_PLACE_HOLDER, "Lorem Ipsum")
        elif key.find("carousel") != -1:
            return value.replace(TEXT_PLACE_HOLDER, "Lorem Carousel")
        elif key.find("text") != -1:
            return value.replace(TEXT_PLACE_HOLDER, "Lorem Ipsum is simply dummy text of the printing and typesetting industry."
                                                    " Lorem Ipsum has been the industry's standard dummy text ever since the 1500s")
        return value

    @staticmethod
    def render_content_with_random_text(key, value):
        if key.find("btn") != -1:
            value = value.replace(TEXT_PLACE_HOLDER, Utils.get_random_text())
        elif key.find("title") != -1:
            value = value.replace(TEXT_PLACE_HOLDER, Utils.get_random_text(length_text=5, space_number=0))
        elif key.find("carousel") != -1:
            buf = value.replace(TEXT_PLACE_HOLDER_1, Utils.get_random_text(length_text=5, space_number=0))
            buf = buf.replace(TEXT_PLACE_HOLDER_2, Utils.get_random_text(length_text=5, space_number=0))
            return buf
        elif key.find("text") != -1:
            value = value.replace(TEXT_PLACE_HOLDER,
                                      Utils.get_random_text(length_text=56, space_number=7, with_upper_case=False))

        return value
