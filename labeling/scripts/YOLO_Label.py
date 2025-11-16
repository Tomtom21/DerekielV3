class YOLO_Label:
    def __init__(self, label_string, classes_dict):
        string_items = label_string.split(" ")
        self.name = classes_dict.get(int(string_items[0]))
        self.x = float(string_items[1])
        self.y = float(string_items[2])
        self.width = float(string_items[3])
        self.height = float(string_items[4])
