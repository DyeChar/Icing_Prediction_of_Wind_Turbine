def bfs(worklist):
    queue =
    while queue =
if __name__ =='__main__':
    n = int(raw_input())
    worklist = []
    for i in range(n):
        temp = str(raw_input())
        list_temp = [False]*6
        for j in temp:
            print int(j)
            list_temp[int(j)] = True
        worklist.append(list_temp)
    count=