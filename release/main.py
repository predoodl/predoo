from readOpJson import readOpInfo
import Op_execute as exe
from CreateCorpus import createCorpus
import numpy as np

def Main():
    op_name =  input("please input the opName: ")
    opInfo = readOpInfo("OpInfo.json", op_name)
    shape = opInfo["InputShape"]
    strategy = input("please select the strategy in random, meanguided, maxguided: ")
    seed_num = int(input("please input the size of corpus: "))
    shape_rd_index = np.random.randint(0,len(shape)-1)
    corpus = createCorpus(seed_num, shape[shape_rd_index])
    if strategy =='random':
        rounds = int(input("please input the number of round: "))
        exe.executeOp_Random(corpus,opInfo,rounds)
    else:
        step = int(input("please input the terminate_num: "))
        exe.execute_guided(corpus, opInfo, step, strategy)




if __name__ == '__main__':
    Main()


