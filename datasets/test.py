import csv
import time
import operator
import datetime
import os
import pickle

with open('sample_train-item-views.csv', "r") as f:
    reader = csv.DictReader(f, delimiter=';')
    sess_clicks = {}  #存储所有点击事件的会话的集合
    sess_date = {}   #存储——{会话ID：会话的日期对应的秒数}，后期用来进行数据拆分（测试集、训练集）
    ctr = 0
    curid = -1
    curdate = None  #再循环内存储事件的日期
    for data in reader:
        sessid = data['session_id']
        if curdate and not curid == sessid:   #当前的id不是session的id，则增加该sessionID的日期转换为秒的结果
            date = ''

            date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            sess_date[curid] = date
        curid = sessid
        item = data['item_id'], int(data['timeframe'])
        curdate = ''

        curdate = data['eventdate']

        if sessid in sess_clicks:   #同一个会话增加点击的商品项目
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    date = ''

    date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
    for i in list(sess_clicks):
        sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
        sess_clicks[i] = [c[0] for c in sorted_clicks]
    sess_date[curid] = date
print("-- Reading data @ %ss" % datetime.datetime.now())


length_berore = len(sess_clicks)
# Filter out length 1 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]
        
# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

#过滤，首先得到去除出现小于5次的项目，然后长度小于2个的会话直接删除，否则替换成去除出现小于5次的项目的点击会话
length=len(sess_clicks)  
for s in list(sess_clicks):   #list(字典)得到的是键的列表
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq
length_after = len(sess_clicks)

print('The number of sessions has changed from %d to %d after filitered' %(length_berore,length_after))

# Split out test set based on dates
dates = list(sess_date.items())
maxdate = dates[0][1]   #Get the second corresponding to the maximum date.

for _, date in dates:
    if maxdate < date:
        maxdate = date

# 7 days for test
splitdate = 0
splitdate = maxdate - 86400 * 7   #A day contains 86400 seconds.

print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
tra_sess = filter(lambda x: x[1] < splitdate, dates)
tes_sess = filter(lambda x: x[1] > splitdate, dates)

# Sort sessions by date
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
print(len(tra_sess))    # 186670    # 7966257
print(len(tes_sess))    # 15979     # 15324
print(tra_sess[:3])
print(tes_sess[:3])
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

item_dict = {}  #The dictionary to record the items
# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    train_ids = []
    train_seqs = []
    train_dates = []
    item_ctr = 1
    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
    print(item_ctr)     # 43098, 37484
    return train_ids, train_dates, train_seqs


# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
    return test_ids, test_dates, test_seqs

tra_ids, tra_dates, tra_seqs = obtian_tra()  #sessionID,to which the data by second corresponding,to which the sequence corresponding.
tes_ids, tes_dates, tes_seqs = obtian_tes()  #test sets wihch are same as above of the contents.

def process_seqs(iseqs, idates):  #每个长度为n的序列拆分成n-1组输入和输出标签
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [id]
    return out_seqs, out_dates, labs, ids

tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
tra = (tr_seqs, tr_labs)
tes = (te_seqs, te_labs)
print(len(tr_seqs))
print(len(te_seqs))
print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])
print(te_seqs[:3], te_dates[:3], te_labs[:3])
all = 0

for seq in tra_seqs:
    all += len(seq)
for seq in tes_seqs:
    all += len(seq)
print('avg length: ', all/(len(tra_seqs) + len(tes_seqs) * 1.0))

if not os.path.exists('sample'):
    os.makedirs('sample')
pickle.dump(tra, open('sample/train.txt', 'wb'))  #所有训练的序列+标签
pickle.dump(tes, open('sample/test.txt', 'wb'))   #所以测试的序列+标签
pickle.dump(tra_seqs, open('sample/all_train_seq.txt', 'wb'))  #所以训练的序列

print('Done.')

