このフォルダについて
---
CNNバージョンの学習及び重みファイルの作成を行う際に使用するスクリプトが置いてあります．
pythonのDeep Learning用ライブラリKerasを用いて学習を行います．


依存関係のインストール
---
このフォルダのスクリプトを実行するにはPython2.7と下記のパッケージをインストールする必要があります．  

 - NumPy (Pythonで科学計算をする際の基礎的なライブラリ)
 - Keras (Deep Learning用ライブラリ)
 - Theano (Kerasのバックエンドとして用いる)
 - h5py (Kerasで学習結果を保存するのに必要)
 - Pillow (画像を扱う際に用いる)

また，KerasのバックエンドとしてTheanoを使用するため  
~/.keras/keras.jsonを開き次の2点を変更します．  

1. "image_dim_ordering"の値を"th"に変更(デフォルトでは"tf")
2. "backend"の値を"theano"に変更(デフォルトでは"tensorflow")

[FaxOCR評価版環境](https://sites.google.com/site/faxocr2010/ji-pc-de-tamesu)ではコマンドラインでこのフォルダに入り  
`$ sudo ./install_packages.sh`  
を実行することで必要なソフトウェアのインストールおよびkeras.jsonの変更が行われます．  


使い方
---
まず，教師データとして用いる画像ファイルを用意し1つのフォルダに格納します．  
画像ファイルはファイル名の先頭の1文字が教師ラベルとして用いられます．  
例えば `9-20120527141018-92.png` であれば `9` が教師ラベルとして用いられます．  
ラベルにはascii文字であれば何でも使うことができますが英数字を使うことを推奨します．  
また，英字の大文字小文字は区別されます．  
**扱うことのできるファイル形式はpng, jpg, pbmのみです**  

画像が準備できたらコマンドラインでこのフォルダに入り  
`$ ./run.sh /path/to/dir/`  
のようにコマンドを実行すると学習が行われます．  
`/path/to/dir/` には教師データを格納したディレクトリへのパスが入ります．

run.shでは他のスクリプトを呼び出し次の3つの処理を行っています．  

1. kocrおよび前処理用プログラムのビルド
2. 教師データの前処理 (make_data.py)
3. CNNの学習 (train_cnn.py)

2.の結果としてimage.npy(画像データ)とlabel.npy(ラベルデータ)が生成されます．  
拡張子の.npyはNumPyで読み書きし易い形式で保存されていることを示しています．  
3.では2.で作成したデータを用いてCNNの学習を行います．  
学習結果としてKerasから使用できるweights.hdf5とkocrから使用できるcnn-result.txtが生成されます．

このcnn-result.txtをkocrの第一引数として設定することでkocrから結果を利用することができます．  
学習後，特に必要ない場合にはimage.npy, label.npy, weights.hdf5は削除して問題ありません．  


CNNのモデル構成の変更方法
---
CNNのモデル構成(レイヤー数やカーネルサイズなど)を変更したい場合には  
このフォルダのtrain_cnn.pyとkocr/src/kocr_cnn.cppの両方を変更しなければいけません．  

train_cnn.pyでは `print "Build model"` 以下のコードでモデルを定義しています．  
Kerasでのモデル定義については[ドキュメント](https://keras.io/getting-started/sequential-model-guide/) [(日本語版)](https://keras.io/ja/getting-started/sequential-model-guide/)を参考にしてください．

kocr_cnn.cppでは関数 `kocr_cnn_init` の中でモデルの定義が行われています．  
モデル自体の実装はkocr/src/forward_cnn.hで行われており，kocr_cnn.cppでは  
KerasのSequentialモデルとほぼ同様に記述するだけでモデルの定義を行うことができます．

例えばKerasを用いてこのように技術したモデルを  

```python
model = Sequential()

model.add(Convolution2D(32, 9, 9, input_shape=(1, nb_dim, nb_dim)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(128, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))
```

C++上で次のように記述することができます．
```c++
net = new Network();

std::vector<int> input_shape(3);
input_shape[0] = 1;
input_shape[1] = 48;
input_shape[2] = 48;

net->add(new Convolution2D(32, 9, 9, input_shape));
net->add(new Relu());
net->add(new MaxPooling2D(2, 2));
net->add(new Dropout(0.5));

net->add(new Convolution2D(64, 5, 5));
net->add(new Relu());
net->add(new MaxPooling2D(2, 2));
net->add(new Dropout(0.5));

net->add(new Convolution2D(128, 3, 3));
net->add(new Relu());
net->add(new MaxPooling2D(2, 2));
net->add(new Dropout(0.5));

net->add(new Flatten());
net->add(new Dense(128));
net->add(new Relu());
net->add(new Dropout(0.5));

net->add(new Dense(nb_classes));
net->add(new Softmax());
```

なお，現在(2016/10/15)C++側で対応しているレイヤーは

 - Dense
 - Convolution2D
 - MaxPooling2D
 - Flatten
 - Dropout
 - Relu
 - Softmax

のみとなります．詳しくはkocr/src/forward_cnn.hをご覧ください．
