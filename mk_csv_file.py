import glob
import pandas as pd
import numpy as np 
import csv 
import os
import pathlib
import re

#パラメータ
#窓の大きさ、カバー率50%、スライドウィンドウでずらす大きさ
win_size = 4
cover_rate = 0.5
sliding_size = int(win_size * cover_rate)

#被験者の体重
subject_weight = [35,55,51]

def main():
    #被験者までのパス
    path2subject = "./raw_data/"
    files = os.listdir(path2subject)
    name_subject_list = [f for f in files if os.path.isdir(os.path.join(path2subject, f))]
    activity_list = ["go_downstairs", "go_upstairs", "stand_down", "stand_up", "walk"]
    sensor_place_list = ["A", "B", "E","Y"]
    
    #activityまで
    path2activity = path2subject + "/"
    
    count_subject = 0
    #被験者毎のループ文
    for name_subject in name_subject_list:
        
        #動作毎のループ文
        for activity in activity_list:
            #センサ毎のループ文
            for sensor_place in sensor_place_list:
                #回数毎のループ文
                #いじっているファイルまでのパス
                path = "{0}/{1}/{2}/".format(name_subject, activity, sensor_place)
                #書き出す用のパス・フォルダを作成
                feature_data_path = "feature_data/"+path
                if os.path.exists(feature_data_path):
                    pass
                else:
                    os.makedirs(feature_data_path)

                num_file = len(glob.glob("raw_data/"+path+"*.csv"))
                print("CSVファイルの数は{}個".format(num_file))
              
                for i in range(num_file):
                    # csv_input = Read_csv(path, i+1)
                    if sensor_place != "Y":                   
                        Feature_extraction_imu(path, i+1, sensor_place,activity,\
                                        feature_data_path+"({})_feature.csv".format(i+1))                           
                    else : 
                        Feature_extraction_force(path, i+1,sensor_place ,activity, count_subject,\
                                        feature_data_path+"({})_feature.csv".format(i+1))
               
                #統合したいファイル名までのパスとnewファイルまでのパスとファイル名
                Marge_csv_raw(feature_data_path, \
                                        "feature_data/{0}/{1}/{2}_feature.csv".format(name_subject, activity, sensor_place))
            #統合したいファイル名までのパスとnewファイルまでのパスとファイル名
            Marge_csv_column("feature_data/{0}/{1}/".format(name_subject, activity), \
                                    "feature_data/{0}/{1}_feature.csv".format(name_subject, activity))
            print("センサ"+sensor_place+"書き込み完了")       
        #統合したいファイル名までのパスとnewファイルまでのパスとファイル名
        Marge_csv_raw("feature_data/{0}/".format(name_subject), \
                        "feature_data/{0}_feature.csv".format(name_subject))
        count_subject += 1 
        print(activity+"書き込み完了")    
    Marge_csv_raw("feature_data/", \
                        "feature_data/all_feature.csv")             
    print("全ての被験者の書き込み完了")


###############################################
#IMUセンサの生データファイルの読み込みと特徴量の計算
###############################################
#Feature_extraction_imu(いじっているファイルまでのパス,i+1番目,センサの位置、動作、 書き込むファイルまでのパス)
def Feature_extraction_imu(path, i, sensor_place,activity,write_file_name): 
    #生データの読み込み
    csv_input = pd.read_csv("raw_data/{}({}).csv".format(path,i), \
                                    skiprows=9, encoding="shift_jis", sep=",")
    #生データの個数
    num_rawdata = len(csv_input.index)
    #何回for文で回すか
    num_for = int(int(num_rawdata/win_size) + int(num_rawdata-sliding_size)/win_size)
    print("for文を{}回回す".format(num_for))

    #特徴量算出、スライドウィンドウでのループ文
    j = 0
    f_quantity_list = []
    for i in range(num_for):
        #平均値算出
        mean_sw = csv_input[j:j+win_size].mean()
        #最大値算出
        max_sw = csv_input[j:j+win_size].max()
        if sensor_place == "A":
            f_quantity_list.append([activity, mean_sw["Acc_X"], mean_sw["Acc_Y"], mean_sw["Acc_Z"], \
            mean_sw["Agl_Vel_X"],mean_sw["Agl_Vel_Y"],mean_sw["Agl_Vel_Z"]])
        else:
            f_quantity_list.append([ mean_sw["Acc_X"], mean_sw["Acc_Y"], mean_sw["Acc_Z"], \
            mean_sw["Agl_Vel_X"],mean_sw["Agl_Vel_Y"],mean_sw["Agl_Vel_Z"]])
        j = j + sliding_size
    #Write_csv(書き込みたいリスト, 書き込むファイル名)
    Write_csv(f_quantity_list,write_file_name, sensor_place)

###################################################
#forceセンサの生データファイルの読み込みと特徴量の計算
###################################################
def Feature_extraction_force(path, i,sensor_place,activity, count_subject, write_file_name):
    #生データの読み込み
    csv_input = pd.read_csv("raw_data/{}({}).csv".format(path,i),usecols=[2],skiprows = 0,encoding="shift_jis", sep=",", header = None)
    #電圧→力→FoBに変換
    function = lambda x:x * 46.331 / subject_weight[count_subject]
    csv_input = csv_input.applymap(function)
    print(csv_input)
    #生データの個数
    num_rawdata = len(csv_input.index)
    print("生データの数は"+str(num_rawdata)+"個")
    #何回for文で回すか
    num_for = int(int(num_rawdata/win_size) + int(num_rawdata-sliding_size)/win_size)
    print("for文を{}回回す".format(num_for))

    #特徴量算出、スライドウィンドウでのループ文
    j = 0
    f_quantity_list = []
    for i in range(num_for):
        #平均値算出
        mean_sw = csv_input[j:j+win_size].mean()
        print(mean_sw)
        #最大値算出
        max_sw = csv_input[j:j+win_size].max()
        # f_quantity_list.append([mean_sw[2]])
        f_quantity_list.append([mean_sw[2]])
        j = j + sliding_size
    #Write_csv(書き込みたいリスト, 書き込むファイル名, センサの場所)
    Write_csv(f_quantity_list,write_file_name, sensor_place)

####################
#データの書き込み
####################
def Write_csv(f_quantity_list,write_file_name, sensor_place):
    #header を特徴量に応じて、書き加える必要がある
    header = []
    if sensor_place == "A":
        header = ["Label","A_AccX_mean","A_AccY_mean","A_AccZ_mean",\
                        "A_AglVelX_mean","A_AglVelY_mean","A_AglVelZ_mean"]
        #書き込むファイル名を引数に受け取る
        with open (write_file_name, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(f_quantity_list)
    elif sensor_place == "B" or sensor_place == "E":
        header = [sensor_place+"_AccX_mean",sensor_place+"_AccY_mean",sensor_place+"_AccZ_mean",\
                        sensor_place+"_AglVelX_mean",sensor_place+"_AglVelY_mean",sensor_place+"_AglVelZ_mean"]
        #書き込むファイル名を引数に受け取る
        with open (write_file_name, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(f_quantity_list)
    elif sensor_place == "Y":
        header = ["FoB"]
        #書き込むファイル名を引数に受け取る
        with open (write_file_name, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(f_quantity_list)


###########################
#ファイルの統合・横につなげる
###########################
#統合したいファイル名までのパスとnewファイルまでのパスとファイル名
def Marge_csv_column(target_files_path, new_file_path):
    csv_files = glob.glob(target_files_path+"*.csv")
    target_files_list = []
    for csv_file in csv_files:
        print(csv_file)
        target_files_list.append(pd.read_csv(csv_file))

    df = pd.concat(target_files_list, axis=1)
    df.to_csv(new_file_path, index=False)


##########################
#ファイルの統合・縦につなげる
##########################
#統合したいファイル名までのパスとnewファイルまでのパスとファイル名
def Marge_csv_raw(target_files_path, new_file_path):
    csv_files = glob.glob(target_files_path+"*.csv")
    target_files_list = []
    for csv_file in csv_files:
        print(csv_file)
        target_files_list.append(pd.read_csv(csv_file))

    df = pd.concat(target_files_list)
    df.to_csv(new_file_path, index=False)

if __name__ == '__main__':
    main()