>《Pytorch 讓你愛不釋手》ISBN:9789863796671
> https://github.com/chenyuntc/pytorch-book


<span id="contents"><a href="#contents"> </span>
* Contents
    * [4. 常用的神經網路層](#ch4)
    * [5. 資料處理](#ch5)
        - [5.3.2 visdom](#ch532)
        - [5.4 GPU和持久化](#ch54)

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

<span id="ch4"></span>
### 4.2 常用的神經網路層
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

<span id="ch5"></span>
### 5. 資料處理
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

<span id="ch532"></span>
#### 5.3.2 visdom
透過pip可以安裝visdom，使用
指令`python -m visdom.server`啟動服務，或是`nohup python -m visdom.server &`指令將服務放至後台執行。visdom服務是一個Web server服務，預設綁定<http://localhost:8097>，用戶與伺服器間透過tornado進行非阻塞互動。繪圖資料支援Pytorch、Numpy，但不支持python的int, float等類型。[p.5-22, 5-24]
```python
%%sh
# 啟動visdom伺服器
# nohup python -m visdom.server &

#新建一個連接客戶端
#指定 env=u'test1'，預設通訊埠為8097，host是'localhost'
vis = visdom.Visdom(env=u'test1') #除了指定env外，還可指定host, port等參數

x = torch.arange(1,30,0.01)
y = torch.sin(x)
vis.line(X=x,Y=y, win='sinx', opts={'title': 'y=sin(x)'})

for ii in range(0,10):
    x1 = torch.Tensor([ii])
    y1 = x1
    #參數update='append'來重疊圖像
    vis = line(x1,y1,win='fig2', update='append' if ii > 0 else None)

#updateTrace 新增一條線
x1 = torch.arange(0,9,0.1)
y1 = x**2 /9
vis.updateTrace(x,y,win='fig2', name='this is a new Trace') 
```
繪圖參數：[p.5-24]
* win：用於指定pane的名子，如果不指定，visdom將自動分配一個新的pane。如果兩次操作指定的win名子一樣，新的操作將覆蓋目前pane的內容。
* opts：用來設定pane的顯示格式，常見的設定包含title、xlabel、ylabel、width等。

[點我返回目錄](#contents)

<span id="ch54"></span>
#### 5.4 GPU和持久化
tensor.cuda()和variable.cuda()的成員方法都會傳回一個新物件，這個新物件的資料以傳輸至GPU，而之前的tensor/variable的資料還在原來的裝置上(CPU)。module.cuda()會將所有的資料都轉移至GPU，並傳回自己。[p.5-28]
在進行低精度的計算時，可以考慮HalfTensor，相比FloatTensor能節省一半的顯示卡記憶體，但務必注意數值溢位的情況。

我們可以透過字典存儲Module和Optimizer物件。[p.5-34]
```python
save_model = dict(
    optimizer = optimizer.state_dict(),
    model = model.state_dict(),
    info = u'模型和優化器的所有參數'
)
torch.save(save_model, 'model.pth')

pretrain = torch.load('./model.pth')
pretrain.keys()
```

[點我返回目錄](#contents)