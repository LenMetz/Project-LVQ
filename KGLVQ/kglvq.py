import numpy as np
import matplotlib.pyplot as plt
import itertools
from numpy import random



class Glvq:
    #https://www.geeksforgeeks.org/self-in-python-class/
    #self refer to the instance of the current class, kinda like this in java

    
    kernel_matrix = np.array([])
    coefficient_vectors = np.array([])

    #about dimension :https://www.youtube.com/watch?v=vN5dAZrS58E&ab_channel=RyanChesler

    def data_normalization(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def vec_normalize(self,arr):
        #arr1 = arr / arr.min()
        arr1 = arr / arr.sum()
        return arr1


    def coeff_initial(self,classnumber,prototype_per_class,samplesize):  #use random numbers to inital the first coefficient vectors of each class
        prototype_number = classnumber * prototype_per_class
        arr = [] #list
        for i in range(0,prototype_number): #coeff vectors for diffrent classes
            for x in range(0, samplesize):
                x = random.rand()
                arr.append(x)   

        arr = np.array(arr)
        arr = np.reshape(arr,(prototype_number,samplesize)) #reshape to a 2d matrix (12,3000)

        self.coefficient_vectors = np.apply_along_axis(self.vec_normalize,1,arr)
            #normalize each coefficient vector , sum up to 1 ,save to class 

        #define prototype labels by order 按0000 1111 2222 的顺序定义 prototype的label, 该顺序永远match coeff vec的所属p的顺序
        prototype_labels = []
        for i in range(0,classnumber):
            for n in range(0,prototype_per_class):
                prototype_labels.append(i) #0000, 1111 ,2222 

        print("p labels: ",prototype_labels)
        return np.array(prototype_labels) #(12,)     








    #gaussian_kernelfunction
    def gaussian_kernelfunction(self,xi,xj):
        sigma = 0.1 #sigma should be changed to fit the data ,0.1
        dist = np.linalg.norm(xi-xj)#euclidean distance
        return np.exp((-(dist)**2)/2*(sigma**2))

        
    def kernelmatrix(self,inputdata):#根据所有数据算出matrix,checked

        matrix = np.array([])#empty 1d array
        all_possible_pairs = list(itertools.product(inputdata, repeat=2)) #重复性配对
        arr = np.array(all_possible_pairs) # convert list to array

        for row in arr:
            paras = np.array([])
            for element in row:#row 里有两个element
                paras = np.append(paras,element)

            newparas = np.reshape(paras,(2,len(inputdata[0]))) #2xn matrix, 因为只有两个vector,len(inputdata[0] to get attribute len
            kernel_result = self.gaussian_kernelfunction(newparas[0],newparas[1])  
            matrix= np.append(matrix,kernel_result)
            
        newmatrix = np.reshape(matrix,(len(inputdata),len(inputdata))) #2d NxN
        self.kernel_matrix=newmatrix #save matrix to class data
        #后面通过坐标[i][j] 即可获得结果, diagonal = 1

    def feature_space_distance_forall_samples(self,prototype_number,samplesize): #每个样本与每个p的距离

        distance_arr = []

        for p in range(0,prototype_number):  #from prototype to sample          
            for index in range(0, samplesize):
                part1 = self.kernel_matrix[index][index] #diagonal = 1
            
                sum1 = 0
                for i in range(0, samplesize):
                    sum1 = sum1 + (self.coefficient_vectors[p][i]*self.kernel_matrix[index][i])

                part2 = sum1

                sum2 = 0
                for s in range(0, samplesize):
                    for t in range(0, samplesize):
                        sum2 = sum2 + (self.coefficient_vectors[p][s]*self.coefficient_vectors[p][t]*self.kernel_matrix[s][t])

                part3 = sum2

                distance =  part1 - (2*part2) + part3
                distance_arr.append(distance)
        distance_arr = np.array(distance_arr)
        distance_arr = np.reshape(distance_arr,(prototype_number,samplesize))
        distance_arr = np.transpose(distance_arr) # 3000x 12

        return distance_arr           

    def feature_space_distance_for_singlesample(self,prototype_number,index,samplesize): #singel样本与每个p的距离

        distance_arr = []

        for p in range(0,prototype_number):  #from prototype to sample          
                part1 = self.kernel_matrix[index][index] #diagonal = 1
            
                sum1 = 0
                for i in range(0, samplesize):
                    sum1 = sum1 + (self.coefficient_vectors[p][i]*self.kernel_matrix[index][i])

                part2 = sum1

                sum2 = 0
                for s in range(0, samplesize):
                    for t in range(0, samplesize):
                        sum2 = sum2 + (self.coefficient_vectors[p][s]*self.coefficient_vectors[p][t]*self.kernel_matrix[s][t])

                part3 = sum2

                distance =  part1 - (2*part2) + part3
                distance_arr.append(distance)
        distance_arr = np.array(distance_arr)


        return distance_arr           



    # define d_plus: 同label的距离
    def distance_plus(self, data_labels, prototype_labels, #checked
                       distance):
        expand_dimension = np.expand_dims(prototype_labels, axis=1) #把prototype label array变成2d, (12,) to (12,1)
        label_transpose = np.transpose(np.equal(expand_dimension, data_labels)) 
        #找到prototype label = data label 的 label, return true/false, data labels中同label的,就被标记为true,
        #(np.equal的结果shape 总是以2d array的row 作为最后的row, 1d array的长度作为最后的column) =>data_labels = (3000,) ; expand_dimension = (12,1)
        #return (12,3000) : 12个prototype 分别跟3000个sample的label做了比较, 所以是12个横行,有3000个ture/false (是否同label)
        # 再transpose成(3000,12) #data现在是横行, prototype是竖行 : 因为euclidean data shape也是(3000,12)每个data到每个prototype的距离
        #transpose后的是按样本序号以此排列的row, euclidean 函数算出的结果也是按sample序号横行排列的

        plus_dist = np.where(label_transpose, distance, np.inf) 
        
        #label_transpose 作为条件, matrix中是true的地方,放入distance,其余都是inf 无效数值
        #=>是否同label的true/false matrix是和distance matrix形状完全吻合的(3000,12), 这样就得到了一个横轴为sample, sample和prototype同label处,才有距离的matrix
        

        d_plus = np.min(plus_dist, axis=1)#(3000,)
        #axis=1:每横轴方向,找到最短距离 => 由于竖轴是12个prototype, 这样就找到了离每个sample最近的,同label的prototye的{距离}, 1d array
        w_plus_index = np.argmin(plus_dist, axis=1) 
        #axis=1:每横轴方向,找到最短距离的元素的[序号],3000个sample中,离每个sample最近的那个p的序号, 1d array 
        #return (3000,)因为每个sample只有一个符合条件的prototype, 该单项数值指代每横行(sample)的12个可能的prototype [0,11]中的某个prototype的序号

        return d_plus, w_plus_index

    # define d_minus: 异label 的距离,checked
    def distance_minus(self, data_labels, prototype_labels,
                        distance):
        expand_dimension = np.expand_dims(prototype_labels, axis=1)
        label_transpose = np.transpose(np.not_equal(expand_dimension, #原理同上,只是not equal,这样就找到每个sample 的不同label的最近的p
                                                    data_labels))

        # distance of non matching prototypes
        minus_dist = np.where(label_transpose, distance, np.inf)
        d_minus = np.min(minus_dist, axis=1)

        # index of minimum distance for non best matching prototypes
        w_minus_index = np.argmin(minus_dist, axis=1)
  
        return d_minus,  w_minus_index

    # define classifier function, <0 为正确分类
    def classifier_function(self, d_plus, d_minus):
        classifier = (d_plus - d_minus) / (d_plus + d_minus) #错误判定公式
        return classifier #(3000,)

    # define sigmoid function
    def sigmoid(self, classifier_result, time_parameter): #xi = ξ
        return 1/(1 + np.exp((-time_parameter) * classifier_result)) #general cost function 中 sum 里的sigmoid公式 (glvq paper)


    def update_ks(self,sample_index,prototype_plus,learning_rate,classifier_result,dk,dl,time_parameter):
        coeff = learning_rate * (self.sigmoid(classifier_result, time_parameter)) * (1 - self.sigmoid(classifier_result,time_parameter))
        #coeff is always the same, from xi to wj
        self.coefficient_vectors[prototype_plus]  =(1 - (coeff * ((4*dl[sample_index])/(dk[sample_index]+dl[sample_index]))))*self.coefficient_vectors[prototype_plus]   #(3000,) normalise all weights for this 
        #self.coefficient_vectors[prototype_plus] => (3000,)
        #PS: here for each weight in vector, the complete list of dk, dl for each data sample is needed, otherwise the classification results will show errors only
        self.coefficient_vectors[prototype_plus][sample_index] = (1 - (coeff * ((4*dl[sample_index])/(dk[sample_index]+dl[sample_index]))))*self.coefficient_vectors[prototype_plus][sample_index]\
            + (coeff * ((4*dl[sample_index])/(dk[sample_index]+dl[sample_index])))
        #self.coefficient_vectors[prototype_plus] = self.vec_normalize(self.coefficient_vectors[prototype_plus])  

        
        #override, single update to right sample's p coeff weight

    def update_kl(self,sample_index,prototype_minus,learning_rate,classifier_result,dk,dl,time_parameter):
        #coeff = learning_rate * (self.sigmoid(classifier_result,time_parameter)) * (1 - self.sigmoid(classifier_result,time_parameter)) #(3000,) coff of all samples
        coeff = learning_rate * (self.sigmoid(classifier_result, time_parameter)) * (1 - self.sigmoid(classifier_result,time_parameter))
        self.coefficient_vectors[prototype_minus]  =(1 + (coeff * ((4*dk[sample_index])/(dk[sample_index]+dl[sample_index]))))*self.coefficient_vectors[prototype_minus]   #(3000,)
        #self.coefficient_vectors[prototype_plus] => (3000,)

        self.coefficient_vectors[prototype_minus][sample_index] = (1 + (coeff * ((4*dk[sample_index])/(dk[sample_index]+dl[sample_index]))))*self.coefficient_vectors[prototype_minus][sample_index]\
            -  (coeff* ((4*dk[sample_index])/(dk[sample_index]+dl[sample_index])))
        #override, single update to right sample's p coeff        
        #self.coefficient_vectors[prototype_minus] = self.vec_normalize(self.coefficient_vectors[prototype_minus])  

        



    # plot  data
    def plot(self, input_data, data_labels, prototypes, prototype_labels):
        plt.scatter(input_data[:, 0], input_data[:, 1], c=data_labels, #0=x ,1=y
                    s=10,cmap='viridis') #cmap = convert data values to rgba
        plt.scatter(prototypes[:, 0], prototypes[:, 1], c=prototype_labels,
                    s=200, marker='P', edgecolor='black',linewidth=2,alpha=0.6)
       
           
    def visualize_2d(self,inputdata):   
        prototype2d = np.dot(self.coefficient_vectors,inputdata)  #(12,3000) * (3000,2) = (12,2), checked
        return prototype2d



    # fit function
    def fit(self, inputdata, data_labels, classnumber,prototype_per_class, learning_rate, epochs):

        input_data = inputdata #normalise?
        samplesize = len(input_data)
        prototype_number = classnumber * prototype_per_class

        prototype_labels = self.coeff_initial(classnumber,prototype_per_class,samplesize)
        
        #initial first coeff vectors for prototypes and their labels
        self.kernelmatrix(input_data)
        #初始化kernel matrix

        distance = self.feature_space_distance_forall_samples(prototype_number,samplesize)
                #ret: (3000,12) 所有sample到所有p的距离,随t改变系数后,更新
        distance_plus, prototype_plus_index = self.distance_plus(data_labels,prototype_labels,distance)
                #ret: 1. 最短的 距离list (3000,) 2.最短距离所属prototype label list (3000,)
        distance_minus, prototype_minus_index = self.distance_minus(data_labels,prototype_labels,distance)
        classifier = self.classifier_function(distance_plus, distance_minus)#if neg, then correct, ret (3000,) 按sample顺序给结果
        #initialize distance and closest distances and classifier results for all samples, 
        #later updates will only update the changes for each single sample, to save calculation time


        cost_function_arr = np.array([])    #cost function array 
        error_count = np.array([])  #error numbers of each iteration
        plt.figure()

        for i in range(epochs): #epochs  

            time_para = 1 # ξ

            for sample_index_t in range(0,samplesize):
                #for index in range(prototype_number):
                #    print("sum of weight vector for prototype {}:".format(index), self.coefficient_vectors[index].sum())
                #更新系数
                distance[sample_index_t] = self.feature_space_distance_for_singlesample(prototype_number,sample_index_t,samplesize) 
                distance_plus[sample_index_t], prototype_plus_index[sample_index_t] = self.distance_plus(data_labels[sample_index_t],prototype_labels,distance[sample_index_t])
                distance_minus[sample_index_t], prototype_minus_index[sample_index_t] = self.distance_minus(data_labels[sample_index_t],prototype_labels,distance[sample_index_t])
                classifier[sample_index_t] = self.classifier_function(distance_plus[sample_index_t], distance_minus[sample_index_t])#if neg, then correct, ret (3000,) 按sample顺序给结果
                #updates for each single sample
                #print("data:{}'s closetest same label prototype:{}, closetest different label prototype:{} ".format(sample_index_t,prototype_plus_index[sample_index_t],prototype_minus_index[sample_index_t]))

                self.update_ks(sample_index_t,prototype_plus_index[sample_index_t],learning_rate,classifier[sample_index_t],distance_plus,distance_minus,time_para)
                #更新系数
                self.update_kl(sample_index_t,prototype_minus_index[sample_index_t],learning_rate,classifier[sample_index_t],distance_plus,distance_minus,time_para)

                
                time_para = 1.0001 * time_para 


            

            cost_function = np.sum(self.sigmoid(classifier,time_para), axis=0) #最终cost function 结果,越小越好

            change_in_cost = 0 #一开始没有change

            if (i == 0):
                change_in_cost = 0

            else:
                change_in_cost = cost_function_arr[-1] - cost_function #cost function结果变化: cost arr 的最后一个element(上次loop中最新的cost function) - 此次cost function结果
                #若结果为正,说明cost function减小了, 就更好

            cost_function_arr = np.append(cost_function_arr, cost_function) #append single cost to cost arr
            print("Epoch : {}, Cost : {} Cost change : {}".format(
                i + 1, cost_function, change_in_cost))

            plt.subplot(1, 2, 1,facecolor='white')
            plt.cla()#so that the updates will not overlap

            
            prototype2d = self.visualize_2d(input_data) #visualize 2d data with the final coeff
            self.plot(input_data, data_labels, prototype2d, prototype_labels) #左图

            
            count  = np.count_nonzero(distance_plus > distance_minus) #如果 d_plus > d_minus 就是判断错误, 因为同label距离应该更近
            error_count = np.append(error_count,count) #每次iteration判断错误的总数
            
            plt.subplot(1, 2, 2,facecolor='black')
            plt.plot(np.arange(i+1), cost_function_arr, marker="d") #右图: 每个cost function的数值变化

            plt.pause(0.1)

        figName = 'KGLVQ_'+ str(samplesize) +'data_samples__'+ str(prototype_number)+'prototypes__' + str(i) + 'epochs' '.png'
        plt.savefig('result/'+figName)
        plt.show()#show last pic
        accuracy = np.count_nonzero(distance_plus < distance_minus) #Counts the number of non-zero values in the array 
        #如果 d_plus < d_minus 就是判断正确, d_plus是正确label,如果距离也是更小,就证明 预判正确
        acc = accuracy / len(distance_plus) * 100 #正确分类占d_plus数量的百分比
        #print(accuracy)
        #print(len(d_plus))
        print("error number per epoch: ",error_count) #每次iteration的错误数
        print("accuracy = {}".format(acc))


  