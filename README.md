>《Pytorch 讓你愛不釋手》ISBN:9789863796671
> https://github.com/chenyuntc/pytorch-book

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