import json

def readOpInfo(filename,opName):
    with open(filename) as f:
        op_list = json.load(f)
    for op in op_list:
        if opName.lower() == op['OpName'].lower():
            return op



# if __name__=="__main__":
#     op = readOpInfo("OpInfo.json", "conv2d")
#     print(op)