import os
import zipfile


def zipHW2(input_path: str, output_path: str):
    zip = zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(os.path.join(input_path, "batch_normalization")):
        fpath = path.replace(input_path, 'HW2')
        for filename in filenames:
            if filename in ["bn.py"]:
                zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    for path, dirnames, filenames in os.walk(os.path.join(input_path, "cifar-10")):
        fpath = path.replace(input_path, 'HW2')
        for filename in filenames:
            if filename in ["dataset.py", "network.py", "train.py"]:
                zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    for path, dirnames, filenames in os.walk(os.path.join(input_path, "results")):
        fpath = path.replace(input_path, 'HW2')
        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    zip.write(os.path.join(input_path, "HW1_BP.py"), os.path.join(input_path.replace(input_path, 'HW2'), "HW1_BP.py"))
    zip.close()


if __name__ == "__main__":

    # ---------------------------------------------------------
    # 请用你的学号和姓名替换下面的内容，注意参照例子的格式，使用拼音而非中文
    id = 21000*****
    name = 'EmptyBlue'
    # ---------------------------------------------------------

    zip_name = f'{id}_{name}.zip'
    input_path = os.path.dirname(__file__)
    output_path = os.path.join(os.path.dirname(__file__), zip_name)

    zipHW2(input_path, output_path)
