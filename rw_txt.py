import os
def read_txt(filename):
    '''
    :param filename:
    :return:
    '''
    with open(filename,"r") as fr:
        fr_line=fr.read()
        line_list=fr_line.strip().split(",")
        indice_list=list(map(int,line_list))
        return indice_list
def write_txt(filename,indice):
    '''
    :param filename:
    :param indice:  每一层保留的特征索引
    :return:
    '''
    sep = ","
    with open(filename, "w") as wf:
        wf.write(sep.join(indice))