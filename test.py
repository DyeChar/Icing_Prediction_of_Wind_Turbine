# coding=utf-8

def transform(n, seq):
    for i in range(n):
        if i+1==n:
            seq[i] = seq[i]+seq[0]
        else:
            seq[i] = seq[i]+seq[i+1]

if __name__== '__main__':
    n, k = map(int,(raw_input().split()))
    sequence = [int(x) for x in raw_input().strip().split()]
    while k>0:
        k -=1
        temp = sequence[1:n]
        temp.append(sequence[0])
        sequence = map(lambda (a,b):a+b,zip(sequence,temp))

    print ' '.join(str(i) for i in sequence)