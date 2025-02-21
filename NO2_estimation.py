import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime
from numpy import pi
from numpy import arctan
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# no2UK=pd.read_csv('D:\\air data\\code\\beijing-air pollution\\London_CODE\\NO2EngRowCol2filtered.csv', index_col=0)
# no2UK['date'] = pd.to_datetime(no2UK['date'])

def today_impu(date1,date2,scalarx,no2UK,indice_tra,columns_to_delete,model_now,year,save_impu=False):
    print(f'now imputation for {date1}')
    predict_X=combineno2_42(no2UK,date1,date2,predict=True)
    predict_X=predict_X.reshape(-1,7,7,42)
    nanP_rows = np.isnan(predict_X).any(axis=(1,2,3))
    cleanedP_matrix = predict_X[~nanP_rows]
    transformed_data = pre_scale_CNN(scalarx,cleanedP_matrix)
    # columns_to_delete = [1, 14, 15, 16, 17, 18, 19, 20, 26]
    transformed_data = np.delete(transformed_data, columns_to_delete, axis=3)
    predictions_flat = model_now.predict([transformed_data,transformed_data[:,3,3,:]])
    keeped_indexes = np.where(~nanP_rows)[0]
    restored_predictions =np.zeros_like(predict_X[:,0,0,0])
    restored_predictions[keeped_indexes] = np.squeeze(predictions_flat)
    img=np.load(f'D:\\air data\\code\\beijing-air pollution\\London_CODE\\CAMS\\CAMS-NO2_24\\{date1}.npy')
    # matrix = np.full_like(img, np.nan)
    matrix = np.full_like(img, np.nan)
    matrix[indice_tra] = np.squeeze(restored_predictions)
    fig = plt.figure(figsize=(10,8))
    plt.imshow(matrix, cmap='jet')
    plt.colorbar()
    plt.show()
    if save_impu:
        np.save(r'D:\air data\code\beijing-air pollution\London_CODE\NO2_estimation\{}\CNN\{}.npy'.format(year,date1),matrix)
    
def this_year_predict(modelpath,date1,date2):
    date1901=date_list(date1,date2,withhyphen=True)
    for today in date1901:
        today_impu(date1,date2,today,modelpath,date_inforn=True)

def pre_scale_RF(feature_scale,predict_data,flag=-16):
    '''
    输入：之前通过downsample 组合的5个group的combine_X 的scaler, 组合的每天的predict_data
    输出：scale的predict_data
    '''
    xy_combine=predict_data.reshape(-1,42)
    scalar_xy=xy_combine[:,:flag]
    # scalerxy.fit(scalar_xy)
    scalarX_transformed = feature_scale.transform(scalar_xy)
    noscalar_xy=xy_combine[:,flag:] # 后面16个不需要做scale
    Xy_transformed=np.concatenate((scalarX_transformed,noscalar_xy), axis=1)
    return Xy_transformed#.reshape(-1,7,7,42)

def pre_scale_CNN(feature_scale,predict_data,flag=-16):
    '''
    输入：之前通过downsample 组合的5个group的combine_X 的scaler, 组合的每天的predict_data
    输出：scale的predict_data
    '''
    xy_combine=predict_data.reshape(-1,42)
    scalar_xy=xy_combine[:,:flag]
    # scalerxy.fit(scalar_xy)
    scalarX_transformed = feature_scale.transform(scalar_xy)
    noscalar_xy=xy_combine[:,flag:] # 后面16个不需要做scale
    Xy_transformed=np.concatenate((scalarX_transformed,noscalar_xy), axis=1)
    return Xy_transformed.reshape(-1,7,7,42)

def pre_no2_today(date1,date2,scalerx,indice_tra,model_now,year,no2UK,h=15, w=12,save_impu=False):
    
    print(f'now imputation for {date1}')
    predict_X=combineno2_42(no2UK,date1,date2,predict=True,single=True)
    nanP_rows = np.isnan(predict_X).any(axis=1)
    #  #模型加载
    # modelh5_path=f'D:\\air data\\code\\beijing-air pollution\\London_CODE\\NO2_estimation\\H5\\cnnr2_{year}_v1.h5'
    cleanedP_matrix = predict_X[~nanP_rows]
    transformed_data = pre_scale_RF(scalerx,cleanedP_matrix)

    predictions_flat = model_now.predict(transformed_data)#[transformed_data,transformed_data[:,3,3,:]]
        
    keeped_indexes = np.where(~nanP_rows)[0]
    
    restored_predictions =np.zeros((predict_X.shape[0],))
    restored_predictions[keeped_indexes] = np.squeeze(predictions_flat)
    
    #print('3 step predicted datasize:',original_data.shape)
    # img=np.load(f'D:\\air data\\code\\beijing-air pollution\\London_CODE\\CAMS\\CAMS-NO2_24\\{date1}.npy')
    # matrix = np.full_like(img, np.nan)
    matrix = np.full((714, 575), np.nan)
    matrix[indice_tra] = np.squeeze(restored_predictions)
    fig = plt.figure(figsize=(h,w))
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar()
    plt.show()
    if save_impu:
        np.save(r'D:\air data\code\beijing-air pollution\London_CODE\NO2_estimation\{}\RF\{}.npy'.format(year,date1),matrix)
    
def restore_nan(predict_row,clean_result):
    # 假设用来预测的样本predict_row包含 NaN 值的矩阵:combined datasize
    # clean_nan 之后用来预测了，结果是clean_result: predicted size
    #现在根据clean_result，以及原始的矩阵进行复原
    # combined_week1 = ...  # 矩阵
    # 找到包含 NaN 值的行索引
    nan_rows = np.isnan(predict_row).any(axis=1)
    # deleted_indexes = np.where(nan_rows)[0]
    keeped_indexes = np.where(~nan_rows)[0]
    # 删除包含 NaN 值的行，并获取清理后的矩阵
    # cleaned_matrix = predict_row[~nan_rows]
    # deleted_indexes = np.where(nan_rows)[0]
    # 复原矩阵
    restored_matrix = predict_row[:,0] # 就是原始NO2 的值
   # restored_matrix =np.zeros_like( predict_row[:,0]) # 这个是和combine大小一样，但是，0值
    
    restored_matrix[keeped_indexes] = np.squeeze(clean_result[:,0])
    #print(restored_matrix.shape)
    return restored_matrix

        
        
def date_list(date1,date2,withhyphen=True):
    from datetime import datetime, timedelta
    if withhyphen:
        start_date = datetime.strptime(date1, '%Y-%m-%d')
        end_date = datetime.strptime(date2, '%Y-%m-%d')
        date_list = []
        current_date = start_date
        while current_date <= end_date:
            date_list.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
    else:
        start_date = datetime.strptime(date1, '%Y-%m-%d')
        end_date = datetime.strptime(date2, '%Y-%m-%d')
        date_list = []
        current_date = start_date
        while current_date <= end_date:
            date_list.append(current_date.strftime('%Y%m%d'))
            current_date += timedelta(days=1)
    return date_list


def plot_feature(cleaned_matrix):
    import seaborn as sns
    new_list=['1-mean_temp','2 min_temp','3 max_temp','4 v10','5 u10','6 ws','7 wd','8 eva','9 preci13','10 preci24'
              ,'11 d2m','12 sp','13 plh','14 rh','15 snowc','16 sde', '17 tcc', 
          '18 cams o3','19 cams tvdo3','20 cams no2','21 cams tvd', '22 impu no2', '23 ndvi sel','24 road_density', '25 dis2road', '26 dis2sea',
          '27 woodland', '28 arable', '29 grassland', '30 mountain', '31 water', '32 built_up_areas', '33 pop',
          '34 ele','35 x','36 y','37 xy','38 x2','39 y2','40 doy','41 dow','42 moy']
    for i in range(cleaned_matrix.shape[1]):       
        # f, ax = plt.subplots()
        # ax.set(title=new_list[i])
        #sns.distplot(cleaned_matrix[:,i], color='blue')
        #sns.histplot(cleaned_matrix[:, i], color='blue', kde=True)
        f=sns.displot(cleaned_matrix[:, i],color='blue', kde=True)
        f.fig.set_size_inches(3, 2)
        f.set(title=new_list[i])
        plt.show()

def restore_nan(predict_row,clean_result):
    # 假设用来预测的样本predict_row包含 NaN 值的矩阵:(none,7,7,42)
    # clean_nan (none-x,7,7,42)之后用来预测了，结果是clean_result: predicted size(none-x,)
    #现在根据clean_result，以及原始的矩阵进行复原
    # 找到包含 NaN 值的行索引
    nan_rows = np.isnan(predict_row).any(axis=(1, 2, 3))
    # deleted_indexes = np.where(nan_rows)[0] 这里删掉的索引就是x的大小
    keeped_indexes = np.where(~nan_rows)[0]
    # 复原矩阵 
    restored_matrix =np.zeros_like(predict_row.shape[0]) # (none,7,7,42)中的none等于每天在boundary里面的indice的数量，恢复和这个数量一样大小的矩阵
    restored_matrix[keeped_indexes] = np.squeeze(clean_result[:,0])
    #print(restored_matrix.shape)
    return restored_matrix

def clean_nan(combined_feature42,no2_y):
    '''
    input x: (none,7,7,42),y: (none,)
    #nan sample: (x,)
    output: cleaned x (none-x,7,7,42), cleaned y (none-x,)
    '''
    combined_feature42=combined_feature42.reshape(-1,7,7,42)
    # 检测包含NaN值的行
    nan_rows = np.isnan(combined_feature42).any(axis=(1, 2, 3))
    # 过滤矩阵，删除包含NaN值的
    cleaned_matrix = combined_feature42[~nan_rows]
    no2_y_without_nan=no2_y[~nan_rows]

    return cleaned_matrix,no2_y_without_nan

def pre_scale(feature_scale,predict_data):
    '''
    输入：之前通过downsample 组合的5个group的combine_X 的scaler, 组合的每天的predict_data
    输出：scale的predict_data
    '''
    xy_combine=predict_data.reshape(-1,42)
    scalar_xy=xy_combine[:,:flag]
    # scalerxy.fit(scalar_xy)
    scalarX_transformed = feature_scale.transform(scalar_xy)
    noscalar_xy=xy_combine[:,flag:] # 后面16个不需要做scale
    Xy_transformed=np.concatenate((scalarX_transformed,noscalar_xy), axis=1)
    return Xy_transformed


def x_y_scalar(xy_combine,flag=-16,forx=True,feature_range=(0, 1),minmax=True):
    '''
    input x: (none,7,7,42)
    #input y: (none,)
    output: standard-x, scalerx 
    '''
    if minmax==True: 
        #print('minmax')
        from sklearn.preprocessing import MinMaxScaler
        scalerxy = MinMaxScaler(feature_range)
    else:
        from sklearn.preprocessing import StandardScaler
        scalerxy = StandardScaler(with_mean=True, with_std=True) 
    if forx:
        xy_combine=xy_combine.reshape(-1,42)
    scalar_xy=xy_combine[:,:flag]
    scalerxy.fit(scalar_xy)
    scalarX_transformed = scalerxy.transform(scalar_xy)
    noscalar_xy=xy_combine[:,flag:] # 后面16个不需要做scale
    Xy_transformed=np.concatenate((scalarX_transformed,noscalar_xy), axis=1)
    return Xy_transformed,scalerxy

# def x_y_scalar(xy_combine,forx=True,feature_range=(0, 1),minmax=True):
#     '''
#     input x: (none,7,7,42), 需要先reshape到二维（none*49,42）
#     #input y: (none,)
#     output: standard-x（none,42）, scalerx 
#     '''
#     if minmax==True: 
#         #print('minmax')
#         from sklearn.preprocessing import MinMaxScaler
#         scalerxy = MinMaxScaler(feature_range)
#     else:
#         from sklearn.preprocessing import StandardScaler
#         scalerxy = StandardScaler(with_mean=True, with_std=True) 
#     if forx:
#         xy_combine=xy_combine.reshape(-1,42)
#     scalerxy.fit(xy_combine)
#     Xy_transformed = scalerxy.transform(xy_combine)
#     return Xy_transformed,scalerxy

def get_train_xy(X,y,train_per=0.7,test_per=0.2,val_per=0.1,y_scale=False):
    '''
    input x: (none,7,7,42)，
    scaled_x: （none*49,42）, train_test_split需要先reshape到二维X.reshape(-1,7,7,42)
    #input y: (none,)
    output: scalerx,train,valid,test
    '''
    X,scalerx= x_y_scalar(X)
    if y_scale:
        print('scale y')
        y,scalery= x_y_scalar(y)
        train_X, X_val, train_y, y_val = train_test_split(X.reshape(-1,7,7,42), y, test_size=(test_per+val_per), random_state=42)
        print('train sample is ',1-test_per-val_per)
        # Split the training set further into training and validation sets
        X_test, X_val, y_test, y_val = train_test_split(X_val, y_val, test_size=(test_per/(test_per+val_per)), random_state=42)
        print('val sample// test sample is  ',test_per/(test_per+val_per))
        print(train_X.shape,X_test.shape,X_val.shape)
        return scalerx,scalery,train_X,X_test, X_val, train_y,y_test, y_val
    else:
        train_X, X_val, train_y, y_val = train_test_split(X.reshape(-1,7,7,42), y, test_size=(test_per+val_per), random_state=42)
        print('train sample is ',1-test_per-val_per)
        # Split the training set further into training and validation sets
        X_test, X_val, y_test, y_val = train_test_split(X_val, y_val, test_size=(test_per/(test_per+val_per)), random_state=42)
        print('val sample// test sample is  ',test_per/(test_per+val_per))
        print(train_X.shape,X_test.shape,X_val.shape)
        return scalerx,train_X,X_test, X_val, train_y,y_test, y_val
    


def daily_feature_npy(npy_path,date,indice_daily,tem=False,ndvi=False,plh=False,impu=False):
    #npy_path=r'D:\air data\code\beijing-air pollution\London_CODE\ERA5_715_13\t2m'
    if plh:
        now_npy_path=os.path.join(npy_path,f'plh_fine_{date}.npy')
        #print(now_npy_path)
        now_npy = np.load(now_npy_path)
        # now_npy_sel=now_npy[indice_daily[:,0],indice_daily[:,1]]
    elif ndvi:
        year=date.split('-')[0]
        mon=date.split('-')[1]
        now_npy_path=os.path.join(npy_path,f'{year}_{mon}_ndvi.npy')
        now_npy=np.load(now_npy_path)
        now_npy=np.squeeze(now_npy)/10000
        now_npy[now_npy < -1] = -1
        now_npy[now_npy > 1] = 1
        # now_npy_sel=now_npy[indice_daily[:,0],indice_daily[:,1]]
    elif tem:  #将max_rep 变成了tem，表示替换最大温度和最小温度
        now_npy_path=os.path.join(npy_path,date+'.npy')
        now_npy=np.load(now_npy_path)
        now_npy[now_npy == np.max(now_npy)] = np.nan # 有那么一些异常值-其实是1e20，表示空缺值
        # now_npy[now_npy > 8*np.std(now_tif)] = np.nan
        # now_npy_sel=now_npy[indice_daily[:,0],indice_daily[:,1]]     
    elif impu:
        now_npy_path=os.path.join(npy_path,f'no2_imputaion_{date}.npy')
        #print(now_npy_path)
        now_npy = np.load(now_npy_path)
        if now_npy.shape[0]==1:
            now_npy= np.squeeze(now_npy)
        # now_npy_sel=now_npy[indice_daily[:,0],indice_daily[:,1]]
    else:
        now_npy_path=os.path.join(npy_path,date+'.npy')    
        now_npy = np.load(now_npy_path)
        
    if now_npy.shape[0]==1:
        now_npy= np.squeeze(now_npy)
        
    now_npy_sel=now_npy[indice_daily[:,0],indice_daily[:,1]]
    
    return now_npy_sel


def rowcol2indices(row_indices,col_indices,window_size=7):
    # 生成窗口偏移量
    offsets = np.arange(-(window_size // 2), window_size // 2 + 1)

    # 生成行和列的偏移量网格
    row_offsets, col_offsets = np.meshgrid(offsets, offsets)

    # 将偏移量网格与行列索引进行组合，得到窗口索引
    row_window = row_indices[:, np.newaxis] + col_offsets.flatten()# 
    col_window = col_indices[:, np.newaxis] + row_offsets.flatten()

    window_indices = np.column_stack((row_window.ravel(), col_window.ravel()))
    return window_indices


def combineno2_sample42(no2UK,group_indexs,predict=False,single=False):
    loaded_dict = np.load(r'/Volumes/Elements SE/HKU air pollution文件/London_CODE/ERA5_Land_nc/ERA5_scale_dict.npy', allow_pickle=True).item()
    loaded_ZRQdict = np.load(r'/Volumes/Elements SE/HKU air pollution文件/London_CODE/ERA5_Land_nc/ERA5_ZQRscale_dict.npy', allow_pickle=True).item()
    loaded_TVDdict = np.load(r'/Volumes/Elements SE/HKU air pollution文件/London_CODE/CAMS/CAMS_tvdscale_dict.npy', allow_pickle=True).item() 
    loaded_PLHdict = np.load( r'/Volumes/Elements SE/HKU air pollution文件/London_CODE/1KM/PLH_scale_dict.npy', allow_pickle=True).item()  

    # surface 7
    ndvi_tif_path=r'/Volumes/Elements SE/HKU air pollution文件/London_CODE/GEE download/NDVI/NDVInpyAll'
    still_combine_path='/Volumes/Elements SE/HKU air pollution文件/London_CODE/1KM/stilll_combined_714t.npy'
    # 温度 3
    tem_mean_path=r'/Volumes/Elements SE/HKU air pollution文件/London_CODE/ERA5_715_13/t2m'
    tem_min_path=r'/Volumes/Elements SE/HKU air pollution文件/London_CODE/ERA5/tasmin_714NPY'
    tem_max_path=r'/Volumes/Elements SE/HKU air pollution文件/London_CODE/ERA5/tasmax_714NPY'

    # 风速，风向，v,u 4
    v10_path=r'/Volumes/Elements SE/HKU air pollution文件/London_CODE/ERA5_715_13/v10'
    u10_path=r'/Volumes/Elements SE/HKU air pollution文件/London_CODE/ERA5_715_13/u10'

    # 气象 10
    eva_path=r'/Volumes/Elements SE/HKU air pollution文件/London_CODE/ERA5_715_13/e'
    preci13_path=r'/Volumes/Elements SE/HKU air pollution文件/London_CODE/ERA5_715_13/tp'#13:00
    preci24_path=r'/Volumes/Elements SE/HKU air pollution文件/London_CODE/ERA5/rainfall_714'
    d2m_path=r'/Volumes/Elements SE/HKU air pollution文件/London_CODE/ERA5_715_13/d2m'
    sp_path=r'/Volumes/Elements SE/HKU air pollution文件/London_CODE/ERA5_715_13/sp'
    plh_path=r'/Volumes/Elements SE/HKU air pollution文件/London_CODE/1KM/correction_plh/plh_f'
    rh_path=r'/Volumes/Elements SE/HKU air pollution文件/London_CODE/ZRQ/r'
    snowc_path=r'/Volumes/Elements SE/HKU air pollution文件/London_CODE/ERA5_715_13/snowc'
    snowd_path=r'/Volumes/Elements SE/HKU air pollution文件/London_CODE/ERA5_715_13/sde'
    tcc_path=r'/Volumes/Elements SE/HKU air pollution文件/London_CODE/ERA5_715_13/tcc'

    # CAMS 4
    cmas_o3_path=r'/Volumes/Elements SE/HKU air pollution文件/London_CODE/CAMS/CAMS_O3_24'
    cmas_tvdo3_path=r'/Volumes/Elements SE/HKU air pollution文件/London_CODE/CAMS/TVD_O3_24'
    cmas_no2_path=r'/Volumes/Elements SE/HKU air pollution文件/London_CODE/CAMS/CAMS-NO2_24'
    cmas_tvdno2_path=r'/Volumes/Elements SE/HKU air pollution文件/London_CODE/CAMS/TVD_NO2_24'

    # imputation 1
    impu_no2_path=r'/Volumes/Elements SE/HKU air pollution文件/London_CODE/GEE download/no2 imputation'#\no2_imputaion_2019-01-01.npy
    # impu_o3_path=r'D:\air data\code\beijing-air pollution\London_CODE\GEE download\O3 imputation'#\O3_imputaion_{}.npy

    # # traffic 3+7
    tra_emiss_combined10_path='/Volumes/Elements SE/HKU air pollution文件/London_CODE/1KM/tra_emiss_combined10_714t.npy'
    
    #boundary
    boundary=np.load(r'/Volumes/Elements SE/HKU air pollution文件/London_CODE/shapefile/England_boundaryfine_1km.npy')
    
    group_no2pd=no2UK.loc[group_indexs]
    
    date_range_hyphen= group_no2pd['date'].unique().strftime('%Y-%m-%d') 
    date_range=[date.replace('-', '') for date in date_range_hyphen] 
    # date_range_hyphen=today#(date1,date2,withhyphen=False) ['2019-01-08', '2019-01-17', '2019-01-23', '2019-01-28', '2019-01-30']
    # date_range=today.replace('-', '')#[date.replace('-', '') for date in date_list]
    # 初始化结果矩阵
    train_features = None
    train_no2 = None
    for i in range(len(date_range)):
        now_day=date_range[i]
        print('now_day',now_day)
        '''
        组装每天station对应的row，col的7*7 矩阵
        '''
        no2UK_today = group_no2pd[group_no2pd['date'].dt.date == pd.to_datetime(date_range[i]).date()]
        today_no2=no2UK_today['no2'].values
        print(f'today sample size {len(no2UK_today)}')
        # print(f'today no2 in total{today_no2.shape}')
        rows = no2UK_today['row'].values
        # print('stations',len(no2UK_today))
        cols = no2UK_today['col'].values
        # 获取（y*49，） 大小的indices
        indices=rowcol2indices(rows,cols,window_size=7)
        # print(f'indices shape {indices.shape}')
        # 温度 3
        tem_m_sel=daily_feature_npy(tem_mean_path,date_range_hyphen[i],indices)#1
        tem_m_sel=tem_m_sel*loaded_dict['t2m'][1]#+loaded_dict['t2m'][2]

        tem_min_sel=daily_feature_npy(tem_min_path,date_range[i],indices,tem=True)#2

        tem_max_sel=daily_feature_npy(tem_max_path,date_range[i],indices,tem=True)#3
        # print(f'tem_max_sel shape {tem_max_sel.shape}')

        #风速，风向 u,v 4
        v10_sel=daily_feature_npy(v10_path,date_range_hyphen[i],indices)#4
        v10_sel=v10_sel*loaded_dict['v10'][1]+loaded_dict['v10'][2]

        u10_sel=daily_feature_npy(u10_path,date_range_hyphen[i],indices)#5
        u10_sel=u10_sel*loaded_dict['u10'][1]+loaded_dict['u10'][2]
        # print(f'u10_sel shape {u10_sel.shape}')
        ws=np.sqrt(u10_sel*u10_sel + v10_sel*v10_sel)#6

        wd=uv2wd(u10_sel,v10_sel)#7

        # 气象 10
        eva_sel=daily_feature_npy(eva_path,date_range_hyphen[i],indices)#8
        eva_sel=eva_sel*loaded_dict['e'][1]+loaded_dict['e'][2]

        preci13_sel=daily_feature_npy(preci13_path,date_range_hyphen[i],indices)#9

        preci24_sel=daily_feature_npy(preci24_path,date_range[i],indices) #10 rainfall 24h-average

        d2m_sel=daily_feature_npy(d2m_path,date_range_hyphen[i],indices)#11
        d2m_sel=d2m_sel*loaded_dict['d2m'][1]#+loaded_dict['d2m'][2]

        sp_sel=daily_feature_npy(sp_path,date_range_hyphen[i],indices)#12
        sp_sel=sp_sel*loaded_dict['sp'][1]+loaded_dict['sp'][2]

        plh_sel=daily_feature_npy(plh_path,date_range_hyphen[i],indices,plh=True)#13
        plh_sel=plh_sel*loaded_PLHdict['plh'][0]+loaded_PLHdict['plh'][1]

        rh_sel=daily_feature_npy(rh_path,date_range_hyphen[i],indices)#14
        rh_sel=rh_sel*loaded_ZRQdict['r'][1]+loaded_ZRQdict['r'][2]

        snowc_sel=daily_feature_npy(snowc_path,date_range_hyphen[i],indices)#15
        snowc_sel=snowc_sel*loaded_dict['snowc'][1]+loaded_dict['snowc'][2]
        snowc_sel[snowc_sel < 0] = 0 
        
        snowd_sel=daily_feature_npy(snowd_path,date_range_hyphen[i],indices)#16
        snowd_sel=snowd_sel*loaded_dict['sde'][1]+loaded_dict['sde'][2]
        snowd_sel[snowd_sel < 0] = 0 
        
        tcc_sel=daily_feature_npy(tcc_path,date_range_hyphen[i],indices)#17

        # CAMS 4
        cmas_o3_sel=daily_feature_npy(cmas_o3_path,date_range_hyphen[i],indices)#18

        cmas_tvdo3_sel=daily_feature_npy(cmas_tvdo3_path,date_range_hyphen[i],indices)#19

        cmas_no2_sel=daily_feature_npy(cmas_no2_path,date_range_hyphen[i],indices)#20

        cmas_tvdno2_sel=daily_feature_npy(cmas_tvdno2_path,date_range_hyphen[i],indices)#21

        # imputation 1
        impu_no2_sel=daily_feature_npy(impu_no2_path,date_range_hyphen[i],indices,impu=True)#22

        #ndvi 1
        ndvi_sel=daily_feature_npy(ndvi_tif_path,date_range_hyphen[i],indices,ndvi=True)#23

        #traffic 3 emission 7
        still_combine10=np.load(tra_emiss_combined10_path)
        now_still10=still_combine10[indices[:,0],indices[:,1],:]#24-33
        # print('now_still10',now_still10.shape)
        
        #surface 6
        still_combine6=np.load(still_combine_path)
        now_still6=still_combine6[indices[:,0],indices[:,1],:]#34-39
        # print('now_still6',now_still6.shape)
        
        # dateinfor3
        combined_dateinfo=dayinmonth(now_day,indices) #40-42（indices,3）
        # print('combined_dateinfo', combined_dateinfo.shape)
        # 开始组装
        combined_matrix_23 = np.column_stack((tem_m_sel, tem_min_sel, tem_max_sel, v10_sel, u10_sel, ws, wd, eva_sel, preci13_sel,
                                              preci24_sel, d2m_sel, sp_sel, plh_sel, rh_sel, snowc_sel, snowd_sel, tcc_sel,
                                             cmas_o3_sel,cmas_tvdo3_sel,cmas_no2_sel,cmas_tvdno2_sel,impu_no2_sel,ndvi_sel))
        # print('combined_matrix_23',combined_matrix_23.shape)
        combined_matrix42 = np.concatenate((combined_matrix_23, combined_dateinfo, now_still10, now_still6), axis=1)

        # 打印组合后的矩阵的形状
        
        print(f'combined_matrix shape should be (station{len(no2UK_today)} * 49 , 0-41)= {combined_matrix42.shape}')

        # 不同日期的叠加
        if train_features is None:
            # print('none')
            train_features = combined_matrix42
            train_no2=today_no2
        else:
            # print('else vstack')
            train_features = np.vstack((train_features, combined_matrix42))
            train_no2 = np.concatenate((train_no2, today_no2))
        
    return train_no2,train_features.reshape(-1,7,7,42)
    
    
def combineno2_42(no2UK,date1,date2,predict=False,single=False):
    loaded_dict = np.load(r'D:\air data\code\beijing-air pollution\London_CODE\ERA5_Land_nc\ERA5_scale_dict.npy', allow_pickle=True).item()
    loaded_ZRQdict = np.load(r'D:\air data\code\beijing-air pollution\London_CODE\ERA5_Land_nc\ERA5_ZQRscale_dict.npy', allow_pickle=True).item()
    loaded_TVDdict = np.load(r'D:\air data\code\beijing-air pollution\London_CODE\CAMS\CAMS_tvdscale_dict.npy', allow_pickle=True).item() 
    loaded_PLHdict = np.load( r'D:\air data\code\beijing-air pollution\London_CODE\1KM\PLH_scale_dict.npy', allow_pickle=True).item()  

    # surface 7
    ndvi_tif_path=r'D:\air data\code\beijing-air pollution\London_CODE\GEE download\NDVI\NDVInpyAll'
    still_combine_path='D:\\air data\\code\\beijing-air pollution\\London_CODE\\1KM\stilll_combined_714t.npy'
    # 温度 3
    tem_mean_path=r'D:\air data\code\beijing-air pollution\London_CODE\ERA5_715_13\t2m'
    tem_min_path=r'D:\air data\code\beijing-air pollution\London_CODE\ERA5\tasmin_714NPY'
    tem_max_path=r'D:\air data\code\beijing-air pollution\London_CODE\ERA5\tasmax_714NPY'

    # 风速，风向，v,u 4
    v10_path=r'D:\air data\code\beijing-air pollution\London_CODE\ERA5_715_13\v10'
    u10_path=r'D:\air data\code\beijing-air pollution\London_CODE\ERA5_715_13\u10'

    # 气象 10
    eva_path=r'D:\air data\code\beijing-air pollution\London_CODE\ERA5_715_13\e'
    preci13_path=r'D:\air data\code\beijing-air pollution\London_CODE\ERA5_715_13\tp'#13:00
    preci24_path=r'D:\air data\code\beijing-air pollution\London_CODE\ERA5\rainfall_714'
    d2m_path=r'D:\air data\code\beijing-air pollution\London_CODE\ERA5_715_13\d2m'
    sp_path=r'D:\air data\code\beijing-air pollution\London_CODE\ERA5_715_13\sp'
    plh_path=r'D:\air data\code\beijing-air pollution\London_CODE/1KM/correction_plh/plh_f'
    rh_path=r'D:\air data\code\beijing-air pollution\London_CODE\ZRQ\r'
    snowc_path=r'D:\air data\code\beijing-air pollution\London_CODE\ERA5_715_13\snowc'
    snowd_path=r'D:\air data\code\beijing-air pollution\London_CODE\ERA5_715_13\sde'
    tcc_path=r'D:\air data\code\beijing-air pollution\London_CODE\ERA5_715_13\tcc'

    # CAMS 4
    cmas_o3_path=r'D:\air data\code\beijing-air pollution\London_CODE\CAMS\CAMS_O3_24'
    cmas_tvdo3_path=r'D:\air data\code\beijing-air pollution\London_CODE\CAMS\TVD_O3_24'
    cmas_no2_path=r'D:\air data\code\beijing-air pollution\London_CODE\CAMS\CAMS-NO2_24'
    cmas_tvdno2_path=r'D:\air data\code\beijing-air pollution\London_CODE\CAMS\TVD_NO2_24'

    # imputation 1
    impu_no2_path=r'D:\air data\code\beijing-air pollution\London_CODE\GEE download\no2 imputation'#\no2_imputaion_2019-01-01.npy
    # impu_o3_path=r'D:\air data\code\beijing-air pollution\London_CODE\GEE download\O3 imputation'#\O3_imputaion_{}.npy

    # # traffic 3+7
    tra_emiss_combined10_path='D:\\air data\\code\\beijing-air pollution\\London_CODE\\1KM\\tra_emiss_combined10_714t.npy'
    
    #boundary
    boundary=np.load(r'D:\air data\code\beijing-air pollution\London_CODE\shapefile\England_boundaryfine_1km.npy')
        
    date_range=date_list(date1,date2,withhyphen=False)
    date_range_hyphen=date_list(date1,date2,withhyphen=True)
    # date_range_hyphen=today#(date1,date2,withhyphen=False) ['2019-01-08', '2019-01-17', '2019-01-23', '2019-01-28', '2019-01-30']
    # date_range=today.replace('-', '')#[date.replace('-', '') for date in date_list]
    # 初始化结果矩阵
    train_features = None
    train_no2 = None
    for i in range(len(date_range)):
        now_day=date_range[i]
        print('now_day',now_day)
        if predict:
            '''
            predict 每天boundary里面的row col 组成的矩阵
            '''
            mask_o3 =(boundary>0)
            indices_mask = np.where(mask_o3) # 这个需要重新组装成
            rows=indices_mask[0]
            cols=indices_mask[1]
            if not single:
                indices=rowcol2indices(rows,cols,window_size=7)
            else:
                indices = np.column_stack((rows, cols))
        else:
            '''
            组装每天station对应的row，col的7*7 矩阵
            '''
            no2UK_today = no2UK[no2UK['date'].dt.date == pd.to_datetime(date_range[i]).date()]
            today_no2=no2UK_today['no2'].values
            # print(f'today no2 in total{today_no2.shape}')
            rows = no2UK_today['row'].values
            # print('stations',len(no2UK_today))
            cols = no2UK_today['col'].values
            # 获取（y*49，） 大小的indices
            indices=rowcol2indices(rows,cols,window_size=7)
        # print(f'indices shape {indices.shape}')
        # 温度 3
        tem_m_sel=daily_feature_npy(tem_mean_path,date_range_hyphen[i],indices)#1
        tem_m_sel=tem_m_sel*loaded_dict['t2m'][1]#+loaded_dict['t2m'][2]

        tem_min_sel=daily_feature_npy(tem_min_path,date_range[i],indices,tem=True)#2

        tem_max_sel=daily_feature_npy(tem_max_path,date_range[i],indices,tem=True)#3
        # print(f'tem_max_sel shape {tem_max_sel.shape}')

        #风速，风向 u,v 4
        v10_sel=daily_feature_npy(v10_path,date_range_hyphen[i],indices)#4
        v10_sel=v10_sel*loaded_dict['v10'][1]+loaded_dict['v10'][2]

        u10_sel=daily_feature_npy(u10_path,date_range_hyphen[i],indices)#5
        u10_sel=u10_sel*loaded_dict['u10'][1]+loaded_dict['u10'][2]
        # print(f'u10_sel shape {u10_sel.shape}')
        ws=np.sqrt(u10_sel*u10_sel + v10_sel*v10_sel)#6

        wd=uv2wd(u10_sel,v10_sel)#7

        # 气象 10
        eva_sel=daily_feature_npy(eva_path,date_range_hyphen[i],indices)#8
        eva_sel=eva_sel*loaded_dict['e'][1]+loaded_dict['e'][2]

        preci13_sel=daily_feature_npy(preci13_path,date_range_hyphen[i],indices)#9

        preci24_sel=daily_feature_npy(preci24_path,date_range[i],indices) #10 rainfall 24h-average

        d2m_sel=daily_feature_npy(d2m_path,date_range_hyphen[i],indices)#11
        d2m_sel=d2m_sel*loaded_dict['d2m'][1]#+loaded_dict['d2m'][2]

        sp_sel=daily_feature_npy(sp_path,date_range_hyphen[i],indices)#12
        sp_sel=sp_sel*loaded_dict['sp'][1]+loaded_dict['sp'][2]

        plh_sel=daily_feature_npy(plh_path,date_range_hyphen[i],indices,plh=True)#13
        plh_sel=plh_sel*loaded_PLHdict['plh'][0]+loaded_PLHdict['plh'][1]

        rh_sel=daily_feature_npy(rh_path,date_range_hyphen[i],indices)#14
        rh_sel=rh_sel*loaded_ZRQdict['r'][1]+loaded_ZRQdict['r'][2]

        snowc_sel=daily_feature_npy(snowc_path,date_range_hyphen[i],indices)#15
        snowc_sel=snowc_sel*loaded_dict['snowc'][1]+loaded_dict['snowc'][2]
        snowc_sel[snowc_sel < 0] = 0 
        
        snowd_sel=daily_feature_npy(snowd_path,date_range_hyphen[i],indices)#16
        snowd_sel=snowd_sel*loaded_dict['sde'][1]+loaded_dict['sde'][2]
        snowd_sel[snowd_sel < 0] = 0 
        
        tcc_sel=daily_feature_npy(tcc_path,date_range_hyphen[i],indices)#17

        # CAMS 4
        cmas_o3_sel=daily_feature_npy(cmas_o3_path,date_range_hyphen[i],indices)#18

        cmas_tvdo3_sel=daily_feature_npy(cmas_tvdo3_path,date_range_hyphen[i],indices)#19

        cmas_no2_sel=daily_feature_npy(cmas_no2_path,date_range_hyphen[i],indices)#20

        cmas_tvdno2_sel=daily_feature_npy(cmas_tvdno2_path,date_range_hyphen[i],indices)#21

        # imputation 1
        impu_no2_sel=daily_feature_npy(impu_no2_path,date_range_hyphen[i],indices,impu=True)#22

        #ndvi 1
        ndvi_sel=daily_feature_npy(ndvi_tif_path,date_range_hyphen[i],indices,ndvi=True)#23

        #traffic 3 emission 7
        still_combine10=np.load(tra_emiss_combined10_path)
        now_still10=still_combine10[indices[:,0],indices[:,1],:]#24-33
        # print('now_still10',now_still10.shape)
        
        #surface 6
        still_combine6=np.load(still_combine_path)
        now_still6=still_combine6[indices[:,0],indices[:,1],:]#34-39
        # print('now_still6',now_still6.shape)
        
        # dateinfor3
        combined_dateinfo=dayinmonth(now_day,indices) #40-42（indices,3）
        # print('combined_dateinfo', combined_dateinfo.shape)
        # 开始组装
        combined_matrix_23 = np.column_stack((tem_m_sel, tem_min_sel, tem_max_sel, v10_sel, u10_sel, ws, wd, eva_sel, preci13_sel,
                                              preci24_sel, d2m_sel, sp_sel, plh_sel, rh_sel, snowc_sel, snowd_sel, tcc_sel,
                                             cmas_o3_sel,cmas_tvdo3_sel,cmas_no2_sel,cmas_tvdno2_sel,impu_no2_sel,ndvi_sel))
        # print('combined_matrix_23',combined_matrix_23.shape)
        combined_matrix42 = np.concatenate((combined_matrix_23, combined_dateinfo,now_still10, now_still6), axis=1)

        # 打印组合后的矩阵的形状
        
        # print(f'combined_matrix shape should be (station{len(no2UK_today)} * 49 , 0-41)= {combined_matrix42.shape}')

        # 不同日期的叠加
        if train_features is None:
            # print('none')
            train_features = combined_matrix42
            if not predict:
                train_no2=today_no2
        else:
            # print('else vstack')
            train_features = np.vstack((train_features, combined_matrix42))
            if not predict:
                train_no2 = np.concatenate((train_no2, today_no2))
    if predict:
        print(f'for daily predict{train_features.shape}')
        return train_features
    else:
        return train_no2,train_features.reshape(-1,7,7,42)

    
def combineo3_42(no2UK,date1,date2,predict=False,single=False):
    loaded_dict = np.load(r'D:\air data\code\beijing-air pollution\London_CODE\ERA5_Land_nc\ERA5_scale_dict.npy', allow_pickle=True).item()
    loaded_ZRQdict = np.load(r'D:\air data\code\beijing-air pollution\London_CODE\ERA5_Land_nc\ERA5_ZQRscale_dict.npy', allow_pickle=True).item()
    loaded_TVDdict = np.load(r'D:\air data\code\beijing-air pollution\London_CODE\CAMS\CAMS_tvdscale_dict.npy', allow_pickle=True).item() 
    loaded_PLHdict = np.load( r'D:\air data\code\beijing-air pollution\London_CODE\1KM\PLH_scale_dict.npy', allow_pickle=True).item()  

    # surface 7
    ndvi_tif_path=r'D:\air data\code\beijing-air pollution\London_CODE\GEE download\NDVI\NDVInpyAll'
    still_combine_path='D:\\air data\\code\\beijing-air pollution\\London_CODE\\1KM\stilll_combined_714t.npy'
    # 温度 3
    tem_mean_path=r'D:\air data\code\beijing-air pollution\London_CODE\ERA5_715_13\t2m'
    tem_min_path=r'D:\air data\code\beijing-air pollution\London_CODE\ERA5\tasmin_714NPY'
    tem_max_path=r'D:\air data\code\beijing-air pollution\London_CODE\ERA5\tasmax_714NPY'

    # 风速，风向，v,u 4
    v10_path=r'D:\air data\code\beijing-air pollution\London_CODE\ERA5_715_13\v10'
    u10_path=r'D:\air data\code\beijing-air pollution\London_CODE\ERA5_715_13\u10'

    # 气象 10
    eva_path=r'D:\air data\code\beijing-air pollution\London_CODE\ERA5_715_13\e'
    preci13_path=r'D:\air data\code\beijing-air pollution\London_CODE\ERA5_715_13\tp'#13:00
    preci24_path=r'D:\air data\code\beijing-air pollution\London_CODE\ERA5\rainfall_714'
    d2m_path=r'D:\air data\code\beijing-air pollution\London_CODE\ERA5_715_13\d2m'
    sp_path=r'D:\air data\code\beijing-air pollution\London_CODE\ERA5_715_13\sp'
    plh_path=r'D:\air data\code\beijing-air pollution\London_CODE/1KM/correction_plh/plh_f'
    rh_path=r'D:\air data\code\beijing-air pollution\London_CODE\ZRQ\r'
    snowc_path=r'D:\air data\code\beijing-air pollution\London_CODE\ERA5_715_13\snowc'
    snowd_path=r'D:\air data\code\beijing-air pollution\London_CODE\ERA5_715_13\sde'
    tcc_path=r'D:\air data\code\beijing-air pollution\London_CODE\ERA5_715_13\tcc'

    # CAMS 4
    cmas_o3_path=r'D:\air data\code\beijing-air pollution\London_CODE\CAMS\CAMS_O3_24'
    cmas_tvdo3_path=r'D:\air data\code\beijing-air pollution\London_CODE\CAMS\TVD_O3_24'
    cmas_no2_path=r'D:\air data\code\beijing-air pollution\London_CODE\CAMS\CAMS-NO2_24'
    cmas_tvdno2_path=r'D:\air data\code\beijing-air pollution\London_CODE\CAMS\TVD_NO2_24'

    # imputation 1
    # impu_no2_path=r'D:\air data\code\beijing-air pollution\London_CODE\GEE download\no2 imputation'#\no2_imputaion_2019-01-01.npy
    impu_o3_path=r'D:\air data\code\beijing-air pollution\London_CODE\GEE download\O3 imputation'#\O3_imputaion_{}.npy

    # # traffic 3+7
    tra_emiss_combined10_path='D:\\air data\\code\\beijing-air pollution\\London_CODE\\1KM\\tra_emiss_combined10_714t.npy'
    
    #boundary
    boundary=np.load(r'D:\air data\code\beijing-air pollution\London_CODE\shapefile\England_boundaryfine_1km.npy')
        
    date_range=date_list(date1,date2,withhyphen=False)
    date_range_hyphen=date_list(date1,date2,withhyphen=True)
    # date_range_hyphen=today#(date1,date2,withhyphen=False) ['2019-01-08', '2019-01-17', '2019-01-23', '2019-01-28', '2019-01-30']
    # date_range=today.replace('-', '')#[date.replace('-', '') for date in date_list]
    # 初始化结果矩阵
    train_features = None
    train_no2 = None
    for i in range(len(date_range)):
        now_day=date_range[i]
        print('now_day',now_day)
        if predict:
            '''
            predict 每天boundary里面的row col 组成的矩阵
            '''
            mask_o3 =(boundary>0)
            indices_mask = np.where(mask_o3) # 这个需要重新组装成
            rows=indices_mask[0]
            cols=indices_mask[1]
            if not single:
                indices=rowcol2indices(rows,cols,window_size=7)
            else:
                indices = np.column_stack((rows, cols))
        else:
            '''
            组装每天station对应的row，col的7*7 矩阵
            '''
            no2UK_today = no2UK[no2UK['date'].dt.date == pd.to_datetime(date_range[i]).date()]
            today_no2=no2UK_today['no2'].values
            # print(f'today no2 in total{today_no2.shape}')
            rows = no2UK_today['row'].values
            # print('stations',len(no2UK_today))
            cols = no2UK_today['col'].values
            # 获取（y*49，） 大小的indices
            indices=rowcol2indices(rows,cols,window_size=7)
        # print(f'indices shape {indices.shape}')
        # 温度 3
        tem_m_sel=daily_feature_npy(tem_mean_path,date_range_hyphen[i],indices)#1
        tem_m_sel=tem_m_sel*loaded_dict['t2m'][1]#+loaded_dict['t2m'][2]

        tem_min_sel=daily_feature_npy(tem_min_path,date_range[i],indices,tem=True)#2

        tem_max_sel=daily_feature_npy(tem_max_path,date_range[i],indices,tem=True)#3
        # print(f'tem_max_sel shape {tem_max_sel.shape}')

        #风速，风向 u,v 4
        v10_sel=daily_feature_npy(v10_path,date_range_hyphen[i],indices)#4
        v10_sel=v10_sel*loaded_dict['v10'][1]+loaded_dict['v10'][2]

        u10_sel=daily_feature_npy(u10_path,date_range_hyphen[i],indices)#5
        u10_sel=u10_sel*loaded_dict['u10'][1]+loaded_dict['u10'][2]
        # print(f'u10_sel shape {u10_sel.shape}')
        ws=np.sqrt(u10_sel*u10_sel + v10_sel*v10_sel)#6

        wd=uv2wd(u10_sel,v10_sel)#7

        # 气象 10
        eva_sel=daily_feature_npy(eva_path,date_range_hyphen[i],indices)#8
        eva_sel=eva_sel*loaded_dict['e'][1]+loaded_dict['e'][2]

        preci13_sel=daily_feature_npy(preci13_path,date_range_hyphen[i],indices)#9

        preci24_sel=daily_feature_npy(preci24_path,date_range[i],indices) #10 rainfall 24h-average

        d2m_sel=daily_feature_npy(d2m_path,date_range_hyphen[i],indices)#11
        d2m_sel=d2m_sel*loaded_dict['d2m'][1]#+loaded_dict['d2m'][2]

        sp_sel=daily_feature_npy(sp_path,date_range_hyphen[i],indices)#12
        sp_sel=sp_sel*loaded_dict['sp'][1]+loaded_dict['sp'][2]

        plh_sel=daily_feature_npy(plh_path,date_range_hyphen[i],indices,plh=True)#13
        plh_sel=plh_sel*loaded_PLHdict['plh'][0]+loaded_PLHdict['plh'][1]

        rh_sel=daily_feature_npy(rh_path,date_range_hyphen[i],indices)#14
        rh_sel=rh_sel*loaded_ZRQdict['r'][1]+loaded_ZRQdict['r'][2]

        snowc_sel=daily_feature_npy(snowc_path,date_range_hyphen[i],indices)#15
        snowc_sel=snowc_sel*loaded_dict['snowc'][1]+loaded_dict['snowc'][2]
        snowc_sel[snowc_sel < 0] = 0 
        
        snowd_sel=daily_feature_npy(snowd_path,date_range_hyphen[i],indices)#16
        snowd_sel=snowd_sel*loaded_dict['sde'][1]+loaded_dict['sde'][2]
        snowd_sel[snowd_sel < 0] = 0 
        
        tcc_sel=daily_feature_npy(tcc_path,date_range_hyphen[i],indices)#17

        # CAMS 4
        cmas_o3_sel=daily_feature_npy(cmas_o3_path,date_range_hyphen[i],indices)#18

        cmas_tvdo3_sel=daily_feature_npy(cmas_tvdo3_path,date_range_hyphen[i],indices)#19

        cmas_no2_sel=daily_feature_npy(cmas_no2_path,date_range_hyphen[i],indices)#20

        cmas_tvdno2_sel=daily_feature_npy(cmas_tvdno2_path,date_range_hyphen[i],indices)#21

        # imputation 1
        impu_o3_sel=daily_feature_npy(impu_o3_path,date_range_hyphen[i],indices,impu=True)#22

        #ndvi 1
        ndvi_sel=daily_feature_npy(ndvi_tif_path,date_range_hyphen[i],indices,ndvi=True)#23

        #traffic 3 emission 7
        still_combine10=np.load(tra_emiss_combined10_path)
        now_still10=still_combine10[indices[:,0],indices[:,1],:]#24-33
        # print('now_still10',now_still10.shape)
        
        #surface 6
        still_combine6=np.load(still_combine_path)
        now_still6=still_combine6[indices[:,0],indices[:,1],:]#34-39
        # print('now_still6',now_still6.shape)
        
        # dateinfor3
        combined_dateinfo=dayinmonth(now_day,indices) #40-42（indices,3）
        # print('combined_dateinfo', combined_dateinfo.shape)
        # 开始组装
        combined_matrix_23 = np.column_stack((tem_m_sel, tem_min_sel, tem_max_sel, v10_sel, u10_sel, ws, wd, eva_sel, preci13_sel,
                                              preci24_sel, d2m_sel, sp_sel, plh_sel, rh_sel, snowc_sel, snowd_sel, tcc_sel,
                                             cmas_o3_sel,cmas_tvdo3_sel,cmas_no2_sel,cmas_tvdno2_sel,impu_no2_sel,ndvi_sel))
        # print('combined_matrix_23',combined_matrix_23.shape)
        combined_matrix42 = np.concatenate((combined_matrix_23, combined_dateinfo,now_still10, now_still6), axis=1)

        # 打印组合后的矩阵的形状
        
        # print(f'combined_matrix shape should be (station{len(no2UK_today)} * 49 , 0-41)= {combined_matrix42.shape}')

        # 不同日期的叠加
        if train_features is None:
            # print('none')
            train_features = combined_matrix42
            if not predict:
                train_no2=today_no2
        else:
            # print('else vstack')
            train_features = np.vstack((train_features, combined_matrix42))
            if not predict:
                train_no2 = np.concatenate((train_no2, today_no2))
    if predict:
        print(f'for daily predict{train_features.shape}')
        return train_features
    else:
        return train_no2,train_features.reshape(-1,7,7,42)
    
def dayinmonth(now_day,indices,yearall=True):
    # now_day = '20190114'
    # indices = 5
    # 返回（indices,3）大小的信息
    # 将日期字符串转换为日期对象
    date_obj = datetime.datetime.strptime(now_day, '%Y%m%d')
    if yearall:
        # 获取日期对象的属性
        day_of_year = date_obj.timetuple().tm_yday  # 获取一年中的第几天
        day_of_week = date_obj.weekday()  # 获取一周中的第几天（0代表星期一）
        month_of_year = date_obj.month  # 获取一年中的第几个月
        # 创建属性矩阵
        attributes = np.array([[day_of_year, day_of_week, month_of_year]])
    else:
        # 提取日期的属性
        day_of_month = date_obj.day
        week_of_month = (date_obj.day - 1) // 7 + 1
        day_of_week = date_obj.weekday()

        # 创建属性矩阵
        attributes = np.array([[day_of_month, week_of_month, day_of_week]])

    # 根据行重复次数扩展属性矩阵
    expanded_attributes = np.repeat(attributes, len(indices[:,0]), axis=0)
    return expanded_attributes


def uv2wd(u,v):
    wd=np.zeros_like(u)
    for i in range(len(u)):
        if u[i] <= 0 and v[i] < 0:
            wd[i] = arctan(u[i] / v[i]) * 180 / pi
        elif u[i] <= 0 and v[i] > 0:
            wd[i] = 180 - arctan(-u[i] / v[i]) * 180 / pi
        elif  u[i] >= 0 and v[i] > 0:
            wd[i] = 180 + arctan(u[i] / v[i]) * 180 / pi
        elif u[i] >= 0 and v[i] < 0:
            wd[i] = 360 - arctan(-u[i] / v[i]) * 180 / pi
        elif u[i] < 0 and v[i] == 0:
            wd[i] = 90
        elif u[i] > 0 and v[i] == 0:
            wd[i] = 270
    return wd

def coeff_determination(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def plot_loss(history):
    import matplotlib.pyplot as plt
    plt.plot(range(len(history.history['loss'])),history.history['loss'],'b',label='Training loss')
    plt.plot(range(len(history.history['loss'])),history.history['val_loss'],'r',label='Validation val_loss')
    plt.title('Traing and Validation mse')
    plt.legend()
    plt.figure()
    plt.plot(range(len(history.history['loss'])),history.history['coeff_determination'],'b',label='Training loss')
    plt.plot(range(len(history.history['loss'])),history.history['val_coeff_determination'],'r',label='Validation val_loss')
    plt.title('Traing and Validation mse')
    plt.legend()
    plt.figure()
    

    
def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 160:
        lr *= 0.5e-1
    elif epoch > 140:
        lr *= 1e-1
    elif epoch > 120:
        lr *= 1e-1
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

# def lr_schedule(epoch):
#     """Learning Rate Schedule

#     Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
#     Called automatically every epoch as part of callbacks during training.

#     # Arguments
#         epoch (int): The number of epochs

#     # Returns
#         lr (float32): learning rate
#     """
#     lr = 1e-3
#     if epoch > 160:
#         lr *= 0.5e-1
#     elif epoch > 140:
#         lr *= 1e-1
#     elif epoch > 100:
#         lr *= 1e-1
#     elif epoch > 60:
#         lr *= 1e-1
#     print('Learning rate: ', lr)
#     return lr

def density_plot(y_test,predict,year_now):
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from scipy.stats import gaussian_kde

    # 计算指标
    r2 = r2_score(y_test.flatten(), predict.flatten())
    rmse = np.sqrt(mean_squared_error(y_test.flatten(), predict.flatten()))
    mae = mean_absolute_error(y_test.flatten(), predict.flatten())

    # 计算点的概率密度
    xy = np.vstack([y_test.flatten(), predict.flatten()])
    z = gaussian_kde(xy)(xy)

    # 排序点，使得密度最高的点最后绘制
    idx = z.argsort()
    y_test, predict, z = y_test.flatten()[idx], predict.flatten()[idx], z[idx]

    # 绘制散点密度图
    fig, ax = plt.subplots()
    plt.scatter(y_test, predict, c=z, s=10, cmap='Spectral_r')
    plt.colorbar(label='Kernal density')

    # 计算拟合曲线
    coefficients = np.polyfit(y_test, predict, 1)
    b, a = coefficients[0], coefficients[1]
    x = np.linspace(min(y_test), max(y_test), 100)
    y_fit = a + b * x


    # 添加拟合曲线和方程
    plt.plot(x, y_fit, color='b')
    plt.plot(x, x, '--', color='b')

    # 添加指标信息
    plt.text(0.05, 0.95, f'R2={r2:.2f}', transform=plt.gca().transAxes, ha='left', va='top')
    plt.text(0.05, 0.90, f'RMSE={rmse:.2f}', transform=plt.gca().transAxes, ha='left', va='top')
    plt.text(0.05, 0.85, f'MAE={mae:.2f}', transform=plt.gca().transAxes, ha='left', va='top')
    plt.text(0.05, 0.80, f'y = {b:.2f}x + {a:.2f}', transform=plt.gca().transAxes, ha='left', va='top')

    # 在图片的右下角添加文本
    text = year_now
    plt.text(0.95, 0.05, text, transform=ax.transAxes,
            fontsize=12, color='black', ha='right', va='bottom')
    # 设置图表标题和坐标轴标签
    # plt.title('Density Scatter Plot with Linear Regression')
    plt.xlabel('Observed NO2 (ug/m3)')
    plt.ylabel('Predicted NO2 (ug/m3)')
    
    # 保存图像
    plt.savefig(f"DensityPlot_{year_now}.png", dpi=600)
    # 显示图例和绘制图表
    # plt.legend()
    plt.show()