def fwrite_af_weight(weight, wfile_name, float_wt_name, fixed_wt_name, MAGNIFICATION):
    import datetime
    import numpy as np
    
    f = open(wfile_name, 'w')
    todaytime = datetime.datetime.today()
    f.write('// '+wfile_name+'\n')
    strdtime = todaytime.strftime("%Y/%m/%d %H:%M:%S")
    f.write('// {0} by marsee\n'.format(strdtime))
    f.write("\n")
    
    f.write('const float '+float_wt_name+'['+str(weight.shape[0])+']['+str(weight.shape[1])+'] = {\n')
    for i in range(weight.shape[0]):
        f.write("\t{")
        for j in range(weight.shape[1]):
            f.write(str(weight[i][j]))
            if (j==weight.shape[1]-1):
                if (i==weight.shape[0]-1):
                    f.write("}\n")
                else:
                    f.write("},\n")
            else:
                f.write(", ")
    f.write("};\n")

    f.write("\n")
    f.write('const ap_fixed<'+str(int(np.log2(MAGNIFICATION))+1)+', 1, AP_TRN_ZERO, AP_SAT> '+fixed_wt_name+'['+str(weight.shape[0])+']['+str(weight.shape[1])+'] = {\n')
    for i in range(weight.shape[0]):
        f.write("\t{")
        for j in range(weight.shape[1]):
            w_int = int(weight[i][j]*MAGNIFICATION+0.5)
            if (w_int > MAGNIFICATION-1):
                w_int = MAGNIFICATION-1
            elif (w_int < -MAGNIFICATION):
                w_int = -MAGNIFICATION
            f.write(str(w_int/MAGNIFICATION))
            if (j==weight.shape[1]-1):
                if(i==weight.shape[0]-1):
                    f.write("}\n")
                else:
                    f.write("},\n")
            else:
                f.write(", ")
    f.write("};\n")

    f.close()

