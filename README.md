Link data gốc:
https://www.dropbox.com/scl/fo/2aczdnx37hxvcfdo4rq4q/AOjRokSTaiKxXmgUyqdcI6k?e=2&preview=Training-Normal-Videos-Part-2.zip&rlkey=5bg7mxxbq46t7aujfch46dlvz&st=2wrec7lu&dl=1


#
P40 bacth: 16, 4.7s/batch, train: 1818 batch => 2.37h
    batch: 30, 8.5s/batch, train: 970 batch => 2.3h ~~same


1.0: Normal -> overfit
2.0: frozen all + MLP
3.0: No frozen, no pretrain, no mlp

Test loss: 0.4028 | Test accuracy: 0.8359 | Precision: 0.6756756756756757 | Recall: 0.03753753753753754 | F1 score: 0.07112375533428165
4.0: frozon half, no mlp