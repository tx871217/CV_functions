import os


class loader():
    # def load_upper_folder(folder):
    #     filenames=[]
    #     for filename in os.listdir(folder):
    #         filenames.append(filename)
    #     return filenames

    # def load_images_from_folder(folder): #loading the folder to read images(storage into a list)
    #     images = []
    #     for filename in os.listdir(folder):
    #         if filename[-4:]==".jpg":
    #             images.append(filename)
    #     return images

    def load_images(path, method):
        if method == "ALL":
            filenames = []
            dir_list = os.listdir(path)
            for i in range(0, len(dir_list)):
                com_path = os.path.join(path, dir_list[i])
                if com_path[-4:] == ".jpg":
                    filenames.append(com_path)
                elif os.path.isdir(com_path):
                    filenames.extend(loader.load_images(com_path, "ALL"))

        elif method == "FOLDER":
            filenames = []
            for file in os.listdir(path):
                if file[-4:] == ".jpg":
                    filenames.append(path+file)

        elif method == "IMAGE":
            filenames = [path]

        return filenames


# print(loader.load_images("C:/Users/tiger/Desktop/20201005 LK/test/test\\","FOLDER"))


# class loader():

#     def load_images(path, method):
#         match method:
#             case "ALL":
#                 filenames = []
#                 dir_list = os.listdir(path)
#                 for i in range(0, len(dir_list)):
#                     com_path = os.path.join(path, dir_list[i])
#                     if com_path[-4:] == ".jpg":
#                         filenames.append(com_path)
#                     elif os.path.isdir(com_path):
#                         filenames.extend(loader.load_images(com_path, "ALL"))
#             case "FOLDER":
#                 filenames = []
#                 for file in os.listdir(path):
#                     if file[-4:] == ".jpg":
#                         filenames.append(path+file)

#             case "IMAGE":
#                 filenames = [path]

#         return filenames
