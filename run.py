import json

with open('../../../3.uc!', 'rb') as f:
    btay = bytearray(f.read())
with open('prussen1.mp3', 'wb') as out:
    for i,j in enumerate(btay):
        btay[i] = j ^ 0xa3
    out.write(bytes(btay))