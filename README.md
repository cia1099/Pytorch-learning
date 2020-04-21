>《Pytorch 讓你愛不釋手》ISBN:9789863796671
> https://github.com/chenyuntc/pytorch-book

<a href="#contents">

* Contents
    * [4. 常用的神經網路層](*ch4)
    * [5. 資料處理](*ch5)

### 遠端存取Jupyter Notebook
首先，開啟IPython，設定密碼，取得加密後的密碼。[p.2-15]
```
from notebook.auto import passwd
passwd() #輸入密碼
Enter password:
Verify password:
'sha1:f9c17b4cc...' #獲得sha1 code
```
新增或編輯隱藏的`~/.jupyter/jupyter_config.py`，輸入以下設定
```python
#加密後的密碼
c.NotebookApp.password = u'sha1:f9c17b4cc...'

# ::綁定所有IP位址，包括IPv4/IPv6的位址
# 如果只想綁定某個ip，改成對應的ip即可
c.NotebookApp.ip = '::'

# 綁定通訊埠號，如果通訊埠已經被佔用，會自動使用下一號通訊埠10000
c.NotebookApp.port = 9999
```
其次，啟動Jupyter Notebook並指定設定檔，輸入以下指令：
```conda
jupyter notebook --config=jupyter_config.py
```
最後，用戶端開啟瀏覽器，存取url <http>http://[伺服器 ip]:9999</http>，輸入密碼，即可存取Jupyter。
若用戶端瀏覽器無法開啟Jupyter，有可能是防火牆的緣故，輸入以下指令開放對應的通訊埠(Linux指令？若使用IPv6，則把指令iptables改成ip6tables)
```
iptables -I INPUT -p tcp --dport 9999 -j ACCEPT
iptables save
```

IPython的常用魔術指令[p.2-10]
|指令|說明|範例|備註|
|---|---|---|---|
|%timeit|檢測某條語句的執行時間|%timeit a.sum()
|%run|執行某個檔案|%run -i a.py|**-i** 選項代表在當前命名空間中執行，此時會使用當前命名空間中的變數，結果也會返回至當前命名空間
|%quickref|顯示快速參考|||
|%who|顯示目前命名空間中的變數|||
|%debug|進入debug模式||按q鍵退出|
|%magic|檢視所有魔術指令|||
|%env|檢視系統環境變量|||
|%xdel|刪除變數並刪除其在IPython上的一切參考|%xdel a|直接釋放 **a** 所指向的空間|
|%hist|查看輸入歷史|||

可以在指令或函式等後面加上"?"或"??"來檢視對應的說明文件或原始程式碼；例如%run?可以檢視它的使用說明，torch.FloatTensor??即可檢視這個類的源碼。

[點我返回目錄](#contents)

<h3 id='ch4'>
### 4.2 常用的神經網路層</h3>
可以直接在網路層類別中，調用成員weight修改參數。[p.4-7]
```python
kernel = torch.ones(3,3)/-9
kernel[1][1] = 1
conv = torch.nn.Conv2d(1,1,(3,3),1, bias=False)
conv.weight.data = kernel.view(1,1,3,3)
# [p.4-8]
bn = nn.BatchNorm1d(4) # 4 channel，初始化標準差為4，平均值為0
bn.weight.data = torch.ones(4)*4
bn.bias.data = t.zeros(4)
```

最佳化器可以設定不同的權重參數，所加成的學習率[p.4-18]
e.g.
```python
class Net(nn.Module):
    def __init__(self):
    super(Net,self).__init__()
    self.features = nn.Sequential(
        nn.Conv2d(3,6,2),
        #...
    )
    self.classifier = nn.Sequential(
        nn.Linear(16*5*5, 120),
        #...
    )
    def forward(self, x):
    o = self.features(x)
    o = o.view(o.size(0), -1)
    o = self.classifier(o)
    return o

net = Net()
# 為不同子網路設定不同的學習率，在finetune中經常用到
# 如果對某個參數不指定學習率，就使用預設學習率
optimizer = optim.SGD([
    {'params': net.features.parameters()}, #學習率為1e-5
    {'params': net.classifier.parameters(), 'lr': 1e-2}
], lr = 1e-5)
```
[點我返回目錄](#contents)

<h3 id='ch5'>
### 5. 資料處理</h3>
實現自訂的資料集需要繼承Dataset，並實現兩個Python魔法方法：[p.5-2]
* \__getitem__：傳回一筆資料或一個樣本。obj[index]相等於obj.\__getitem__(index)
* \__len__：傳回樣本數量。len(obj)相等於obj.\__len__()

torchvision.transform.Compose類似nn.Sequential，可以將一連串的影像前處理類串聯起來。[p.5-4]

torchvision.datasets.ImageFolder，它假設所有的育訓練集合依資料夾做分類，每個資料夾下儲存同一類別的圖片，資料夾名為類別名。[p.5-6]
```python
!tree --charset ASCII filePath #顯示filePath下的子資料夾樹狀檔結構
dataset = ImageFolder('filePath')
dataset.class_to_idx #檢查資料夾名和label的對應關係，label預設按資料夾名順序排序後存成字典
datasets.imgs #所有圖片的路徑和對應的label
```

torch.utils.data.sampler.WeightedRandomSampler，會根據每個樣本的加權選取資料，在樣本比例不均勻的問題中，可用它進行重取樣。[p.5-13]
```python
weights = [2 if label == 1 else 1 for _, label in dataset]
# replacement用於指定是否可以重複選取樣本
# num_samples為一個Epoche共選取的樣本總數
sampler = WeightedRandomSampler(weights,\
 num_samples=len(dataset), replacement=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,sampler=sampler)
```
如果指定了sampler，shuffle將不再生效，並且sampler.num_samples會覆蓋dataset的實際資料量大小。

[點我返回目錄](#contents)

