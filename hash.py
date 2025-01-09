import hashlib

result = hashlib.md5(b'a')

str = b'a'

for i in range(1000000):
    if i%1000 == 0:
        print(i)
    str += b"a" * i

    hash = hashlib.md5(str)
    if hash.hexdigest() == result.hexdigest():
        print("koniec", i)