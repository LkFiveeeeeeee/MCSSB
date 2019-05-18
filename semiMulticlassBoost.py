import numpy as np
from sklearn import neighbors
from sklearn.svm import SVC
import xgboost
from scipy import sparse
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import OneHotEncoder


def turnYtoOneHot(t_array):
    t_array = np.reshape(t_array, (-1, 1))
    enc = OneHotEncoder()
    enc.fit(np.array([0,1,2,3,4]).reshape(-1,1))
    t_array = enc.transform(t_array).toarray()
    return t_array


def turnOneHotToY(a):
    a = np.argmax(a, axis=1)
    return a


class SemiMultiBoostClassifier:

    def __init__(self, base_model=SVC()):

        self.BaseModel = base_model

    ## b shape(class_num,full_data)
    def _computeLb(self,H):
        b = np.zeros((self.class_num,H.shape[0]))
        H_e = np.exp(H)
        sum = np.sum(H_e,axis=1)
        for i in range(self.class_num):
            b[i] = np.true_divide(H_e[:,i],sum)
        return b

    # Zu shape(unlabeled_length,unlabeled_length)
    def _computeZu(self, b):
        bu = b[:, self.label_length:]   #shape(5,unlabeled_data)
        return np.dot(bu.T, bu)

    # tao shape(class_num,unlabeled_length,unlabeled_length)
    def _computeTao(self,b,Zu):
        bu = b[:,self.label_length:]
        tao = np.zeros((self.class_num, bu.shape[1], bu.shape[1]))
        for i in range(self.class_num):
            b_temp = np.reshape(bu[i], (1, -1))
            tao[i] = np.dot(b_temp.T, b_temp)
            tao[i] = np.true_divide(tao[i], Zu)

        return tao

    # tao alpha(class_num,unlabeled_length)
    def _computeAlpha(self,b,tao,Zu):
        bu = b[:, self.label_length:]
        Su = self.S[self.label_length:,self.label_length:]
        alpha = np.zeros((self.class_num,bu.shape[1]))
        for i in range(self.class_num):
            Zss = bu[i] - tao[i]
            dive_part = np.true_divide(Su, Zu)
            temp_result = np.multiply(Zss, dive_part)
            sum_value = np.sum(temp_result, axis=0)
            alpha[i] = sum_value

        return alpha

    # warning  may false!!!!!  === check sigma symbol
    # fai shape(class_num,labeled_length,unlabeled_length)
    def _computeFai(self,b,H):
        fai = np.zeros((self.class_num,self.label_length,self.unlabel_length))
        bu = b[:, self.label_length:]
        Hl = H[:self.label_length, :]
        Hu = H[self.label_length:, :]
        bu_sum = np.sum(bu,axis=0)
        for i in range(self.class_num):
            bu_divide = bu[i]/bu_sum
            bu_divide = bu_divide.reshape(1,-1)
            Hu_divide = Hu[:, i]/bu[i]
            Hl_temp = np.reshape(Hl[:,i],(-1,1))
            Hl_dot = np.dot(Hl_temp,bu_divide)
            fai[i] = Hl_dot-Hu_divide
        return fai

    # beta shape(class_num,unlabeled_length)
    def _computeBeta(self,fai):
        Sul = self.S[:self.label_length, self.label_length:]
        beta = np.zeros((self.class_num, self.unlabel_length))
        for i in range(self.class_num):
            S_fai_multi = np.multiply(Sul,fai[i])
            S_fai_multi = np.sum(S_fai_multi,axis=0)
            beta[i] = S_fai_multi
        return beta

    # weight shape(unlabeled_length,)
    def _computeWeight(self,k,sArray):
        weight = np.zeros(self.unlabel_length)
        for i in range(len(k)):
            if sArray[k[i],i] >= 0:
                weight[i] = sArray[k[i],i]
            else:
                weight[i] = 0

        return weight

    # Au shape(unlabeled_length,unlabeled_length)
    def _computeAu(self,Zu,b,h):
        bu = b[:, self.label_length:]
        Su = self.S[self.label_length:, self.label_length:]
        h_b = h*(bu.T)
        h_b = np.sum(h_b,axis=1)
        S_Z_divide = Su/Zu
        S_Z_divide = np.sum(S_Z_divide,axis=1)
        Au = np.dot(S_Z_divide.T,h_b)
        return Au

    def _computeAl(self,H,h,b):
        bu = b[:, self.label_length:]
        Hl = H[:self.label_length,:]
        Slu = self.S[:self.label_length,self.label_length:]
        h_b = h * (bu.T)
        h_b = np.sum(h_b, axis=1)
        y_i_b_j_k = np.zeros((self.class_num,self.label_length,self.unlabel_length))
        for i in range(self.class_num):
            y_i_b_j_k[i] = Hl[:,i].reshape(-1,1)/bu[i,:].reshape(1,-1)
        h_b = h_b.reshape(1,-1)
        y_i_b_j = np.sum(y_i_b_j_k,axis=0)
        mul_value = Slu*y_i_b_j*h_b
        Al = np.sum(mul_value)/2.0

        return Al

    def _computeBu(self,h,tao,Zu):
        Su = self.S[self.label_length:, self.label_length:]
        S_Z_divide = Su/Zu
        temp = np.zeros((self.class_num,self.unlabel_length,self.unlabel_length))
        for i in range(self.class_num):
            temp_h = h[:,i].reshape(-1,1)
            temp_tao = tao[i]
            temp[i] = temp_h * temp_tao
        temp_k_sum = np.sum(temp,axis=0)
        all_multi = np.multiply(S_Z_divide,temp_k_sum)
        Bu = np.sum(all_multi)
        return Bu

    def _computeBl(self,H,h,b):
        Hl = H[:self.label_length, :]
        bu = b[:, self.label_length:]
        Slu = self.S[:self.label_length,self.label_length:]
        h_b_divide = h.T/bu
        temp_P =  np.zeros((self.class_num,self.label_length,self   .unlabel_length))
        for i in range(self.class_num):
            Hl_i = Hl[:,i].reshape(-1,1)
            h_b_divide_i = h_b_divide[i].reshape(1,-1)
            temp_P[i] = np.dot(Hl_i,h_b_divide_i)

        temp_P_sum_k = np.sum(temp_P,axis=0)
        all_multi = np.multiply(Slu,temp_P_sum_k)
        Bl = np.sum(all_multi)/2.0
        return Bl


    def fit(self, X,y, unlabeled, C=10000,
            n_neighbors=4, n_jobs=1,
            max_models=20,
            class_num = 5,
            sample_percent=0.1,
            sigma_percentile=95,
            similarity_kernel='rbf',
            verbose=True):
        ''' Fit model'''
        # Localize labeled data
        labeled_data_X = X.values
        labeled_data_Y = y.values
        unlabeled_data = unlabeled.values
        self.class_num = class_num
        S_data_X = np.concatenate((labeled_data_X,unlabeled_data),axis=0)



        # First we need to create the similarity matrix
        if similarity_kernel == 'knn':

            self.S = neighbors.kneighbors_graph(S_data_X,
                                                n_neighbors=n_neighbors,
                                                mode='distance',
                                                include_self=True,
                                                n_jobs=n_jobs)

            self.S = sparse.csr_matrix(self.S)

        elif similarity_kernel == 'rbf':
            # First aprox
            self.S = np.sqrt(rbf_kernel(S_data_X,S_data_X, gamma=1))
            # set gamma parameter as the 15th percentile
            # sigma = np.percentile(np.log(self.S), sigma_percentile)
            # sigma_2 = (1 / sigma ** 2) * np.ones((self.S.shape[0], self.S.shape[0]))
            # self.S = np.power(self.S, sigma_2)
            # # Matrix to sparse
            # self.S = sparse.csr_matrix(self.S)

        else:
            print('No kernel type ', similarity_kernel)

        # =============================================================
        # Initialise variables
        # =============================================================
        self.models = []
        self.weights = []
        H_L = turnYtoOneHot(labeled_data_Y)
        H_U = self.BaseModel.predict_proba(unlabeled_data)
        H = np.concatenate((H_L,H_U),axis=0)

        # Loop for adding sequential models
        for t in range(max_models):
            #=====================================================
            # compute u and l
            #=====================================================
            self.label_length = labeled_data_X.shape[0]
            self.unlabel_length = unlabeled_data.shape[0]
            b = self._computeLb(H)
            Zu = self._computeZu(b)
            tao = self._computeTao(b, Zu)
            alpha = self._computeAlpha(b, tao, Zu)
            fai = self._computeFai(b,H)
            beta = self._computeBeta(fai)/2
#            print("alpha ",alpha,"beta ",beta)
            #=====================================================
            # choose suitable k
            #=====================================================
            sample_array = alpha + C*beta
            sui_k = np.argmax(sample_array,axis=0)
            weight = self._computeWeight(sui_k,sample_array)
            weight_prob = weight/ weight.sum()
            # =============================================================
            # Sample sample_percent most confident predictions
            # =============================================================
            length = self.unlabel_length - len(np.where(weight_prob == 0)[0])
            if length < sample_percent * self.unlabel_length:
                print("0 is fewer than arranged selected num")
                break
            idx_u = np.random.choice(np.arange(self.unlabel_length),
                                       size=int(sample_percent * self.unlabel_length),
                                       p=weight_prob,
                                       replace=False)

            select_sample = unlabeled_data[idx_u]
            select_sample_y = sui_k[idx_u]
            #===============================================================
            # Create new dataSet
            #===============================================================
            new_label_data_x = np.concatenate((labeled_data_X,select_sample),axis=0)
            new_label_data_y = np.concatenate((labeled_data_Y,select_sample_y),axis=0)


            # =============================================================
            # Fit BaseModel to samples using predicted labels
            # =============================================================
            # Fit model to unlabeled observations
            clf = self.BaseModel
 #           print(self.BaseModel)
            clf.fit(new_label_data_x, new_label_data_y)
            # Make predictions for unlabeled observations
            h = clf.predict(unlabeled_data)   # shape may = (unl,)
            h = turnYtoOneHot(h)
            # ==============================================================
            # Compute weight(a)
            # ==============================================================
            Au = self._computeAu(Zu,b,h)
            Al = self._computeAl(H,h,b)
            Bu = self._computeBu(h,tao,Zu)
            Bl = self._computeBl(H,h,b)
            print(str(Au), " ", str(Al), " ", str(Bu), " ", str(Bl))
            a = 0.25*np.log((Bu+C*Bl)/(Au+C*Al))

            # ==============================================================
            # Refresh labeled and unlabeled array
            # ==============================================================
            labeled_data_X = new_label_data_x
            labeled_data_Y = new_label_data_y
            self.label_length = labeled_data_X.shape[0]
            new_unlabeled_data = np.delete(unlabeled_data,idx_u,axis=0)
            unlabeled_data = new_unlabeled_data
            self.unlabel_length = unlabeled_data.shape[0]

            if verbose:
                print('There are still ', self.unlabel_length, ' unlabeled observations')
                print('in' + str(t) + 'times a is' + str(a))


            # Update final model
            # =============================================================
            # If a<0 the model is not converging
            if a < 0:
                if verbose:
                    print('Problematic convergence of the model. a<0')
                break

            # Save model
            self.models.append(clf)
            # save weights
            self.weights.append(a)
            # Update
            H_U = np.zeros((self.unlabel_length,self.class_num))
            w = np.sum(self.weights)
            H_L = turnYtoOneHot(labeled_data_Y)
            for i in range(len(self.models)):
                H_U = np.add(H_U, self.weights[i] * self.models[i].predict_proba(unlabeled_data))
            H = np.concatenate((H_L,H_U),axis=0)

            # =============================================================
            # Breaking conditions
            # =============================================================

            # Maximum number of models reached
            if (t == max_models) & verbose:
                print('Maximum number of models reached')

            # If no samples are left without label, break
            if unlabeled_data.shape[0] == 0:
                if verbose:
                    print('All observations have been labeled')
                    print('Number of iterations: ', t + 1)
                break

        if verbose:
            print('\n The model weights are \n')
            print(self.weights)

    def predict(self, X):
        estimate = np.zeros((X.shape[0],self.class_num))
        print(self.weights)
        print(self.models)
        # Predict weighting each model
        w = np.sum(self.weights)
        for i in range(len(self.models)):
            # estimate = np.add(estimate,  self.weights[i]*self.models[i].predict_proba(X)[:,1]/w)
            estimate = np.add(estimate, self.weights[i] * self.models[i].predict_proba(X))
        result = np.argmax(estimate,axis=1)
        return result
