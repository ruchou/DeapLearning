array=[1,-2,3,0,23,2,0]

tmp_array=list()

num_of_zeros=0

for i in array:
    if i==0:
        num_of_zeros+=1
    else:
        tmp_array.append(i)

for i in range(num_of_zeros):
    tmp_array.append(0)

array=tmp_array
print(array)
        