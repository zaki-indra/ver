class BaseException(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class DirectoryError(BaseException):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class FileError(BaseException):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)