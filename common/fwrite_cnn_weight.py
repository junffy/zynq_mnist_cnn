# This code is based on 
# https://github.com/oreilly-japan/deep-learning-from-scratch
# http://marsee101.blog19.fc2.com
# This modified code also takes over the MIT License.

def fwrite_cnn_weight(weight, wfile_name, float_wt_name, fixed_wt_name, MAGNIFICATION):
    import datetime
    import numpy as np
    
    f = open(wfile_name, 'w')
    todaytime = datetime.datetime.today()
    f.write('// '+wfile_name+'\n')
    strdtime = todaytime.strftime("%Y/%m/%d %H:%M:%S")
    f.write('// {0} by marsee\n'.format(strdtime))
    f.write("\n")
    
    f.write('const float '+float_wt_name+'['+str(weight.shape[0])+']['+str(weight.shape[1])+']['+str(weight.shape[2])+']['+str(weight.shape[3])+'] = \n{\n')
    for i in range(weight.shape[0]):
        f.write("\t{\n")
        for j in range(weight.shape[1]):
            f.write("\t\t{\n")
            for k in range(weight.shape[2]):
                f.write("\t\t\t{")
                for m in range(weight.shape[3]):
                    f.write(str(weight[i][j][k][m]))
                    if (m==weight.shape[3]-1):
                        f.write("}")
                    else:
                        f.write(",")
                
                if (k==weight.shape[2]-1):
                    f.write("\n\t\t}\n")
                else:
                    f.write(",\n")

            if (j==weight.shape[1]-1):
                f.write("\t}\n")
            else:
                f.write(",\n")
        
        
        if (i==weight.shape[0]-1):
            f.write("};\n")
        else:
            f.write("\t,\n")

    f.write("\n")
    f.write('const ap_fixed<'+str(int(np.log2(MAGNIFICATION))+1)+', 1, AP_TRN_ZERO, AP_SAT> '+fixed_wt_name+'['+str(weight.shape[0])+']['+str(weight.shape[1])+']['+str(weight.shape[2])+']['+str(weight.shape[3])+'] = \n{\n')
    for i in range(weight.shape[0]):
        f.write("\t{\n")
        for j in range(weight.shape[1]):
            f.write("\t\t{\n")
            for k in range(weight.shape[2]):
                f.write("\t\t\t{")
                for m in range(weight.shape[3]):
                    w_int = int(weight[i][j][k][m]*MAGNIFICATION+0.5)
                    if (w_int > MAGNIFICATION-1):
                        w_int = MAGNIFICATION-1
                    elif (w_int < -MAGNIFICATION):
                        w_int = -MAGNIFICATION
                    f.write(str(w_int/MAGNIFICATION))
                    if (m==weight.shape[3]-1):
                        f.write("}")
                    else:
                        f.write(",")
                
                if (k==weight.shape[2]-1):
                    f.write("\n\t\t}\n")
                else:
                     f.write(",\n")

            if (j==weight.shape[1]-1):
                f.write("\t}\n")
            else:
                f.write(",\n")
        
        
        if (i==weight.shape[0]-1):
            f.write("};\n")
        else:
            f.write("\t,\n")
 
    f.close()
