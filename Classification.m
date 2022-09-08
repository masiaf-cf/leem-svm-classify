function Classification(sf,max_sigma)
%Reference:Low-energy electron microscopy intensity-voltage data --
%factorization, sparse sampling, and classification"
%Francesco Masia, Wolfgang Langbein, Simon Fischer, Jon-Olaf Krisponeit,
%Jens Falta
%https://arxiv.org/abs/2203.12353

%This function classifies the hyperspectral images according to the method
%described in the paper

%sf: threshold value for sigma_tilde identify low confidence classification
%max_sigma: defines the maximum value of sigma used in the visualisation

D=double(readtiff('PrOx.tif'));%LEEM Data - full spectral information
D=reshape(D,[],size(D,3));
Test_C=double(readtiff('PrOx_FSC3_C.tif'));%LEEM FSC3 Concentration for Training and Testing
l=size(Test_C,1);%length of image
w=size(Test_C,2);%width of image
Test_C=reshape(Test_C,[],size(Test_C,3));
NComp=size(Test_C,2);
NSpec=size(D,2);
en=7:0.1:40;

Mask{1}=~imread('Mask_1.tif');%Masks to define Training ROIs
Mask{2}=~imread('Mask_2.tif');
Mask{3}=~imread('Mask_3.tif');
Mask{4}=~imread('Mask_4.tif');
Mask{5}=~imread('Mask_5.tif');
Mask{6}=~imread('Mask_6.tif');
Mask{7}=~imread('Mask_7.tif');

NClasses=7;
Train_C=cell(1,NClasses);%Defining Training populations
for i=1:NClasses
    Train_C{i}=Test_C(Mask{i}(:),:);
end

NTr=zeros(1,NClasses);%Number of objects in the training sets
for i=1:NClasses
    NTr(i)=size(Train_C{i},1);
end
NTe=size(Test_C,1);

MeanStd_C_Train=zeros(2*NClasses,NComp);%Mean value of features in the training regions
for i=1:NClasses
    MeanStd_C_Train(2*i-1,:)=mean(Train_C{i},1);
    MeanStd_C_Train(2*i,:)=std(Train_C{i},1);
end
dlmwrite('PrOx_FSC3_C_Tr.dat',cat(1,1:NComp,MeanStd_C_Train)');

for i=1:length(sf)
    %Train svm model and return classification of Test data 
    [~,Te.full]=calc_class(NTe,NClasses,Train_C,NTr,Test_C,sf(i));
    Im=createImage(Te.full);
    imwrite(Im,['PrOx_Class_sf' num2str(sf(i)) '_maxSigma' num2str(max_sigma) '.tif']);
    
    Mean_S_Class=zeros(NClasses,NSpec);%Mean value of features in the classification regions
    MeanStd_S_Class=zeros(2*NClasses,NSpec);%Mean value +/- std of features in the classification regions
    
    for j=1:NClasses
        Mean_S_Class(j,:)=mean(D(Te.full.Id==j&~Te.full.lc,:),1);
        Std_S_Class=std(D(Te.full.Id==j&~Te.full.lc,:),1,1);
        MeanStd_S_Class(2*j,:)=Mean_S_Class(j,:)-Std_S_Class;
        MeanStd_S_Class(2*j-1,:)=Mean_S_Class(j,:)+Std_S_Class;
    end
    
    dlmwrite('PrOx_S_Class_Mean.dat',cat(1,en,Mean_S_Class)');
    dlmwrite('PrOx_S_Class_Range.dat',cat(1,en,MeanStd_S_Class)');
end


    function [svmmdl,Te]=calc_class(NT,NC,TrainC,NTr,TestC,sf)
        %Trains a classifier svmmdl using training set Train C and apply to
        %test data TestC
        
        %Input:
        %NT: number of objects in the test set
        %NC: number of classes
        %TrainC: 1 x Nc class defining the training set
        %NTr: array giving the number of objects in the training set
        %sf: threshold value for sigma_tilde identify low confidence classification

        %Output:
        %svmmdl: classification model
        %Te: structure containing the results of the classification on the test set
        %using the full spectral information, (Te.full.Id: 1 x p array containing
        %the class identifier, Te.full.st: 1 x p array containing the value of
        %sigma_tilde as defined in the paper, Te.full.lc: 1 x p logical array
        %showing if the sigma_tilde is above the threshold sf)
        combs=nchoosek(1:NC,2);
        AbsProb=ones(NT,NC);
        ProbNorm=ones(NT,NC);
        mdl=cell(1,size(combs,1));
        m1=zeros(size(combs,1),1);
        s1=zeros(size(combs,1),1);
        m2=zeros(size(combs,1),1);
        s2=zeros(size(combs,1),1);
        
        for k=1:size(combs,1)
            i1=combs(k,1);
            i2=combs(k,2);
            [m1(k),m2(k),s1(k),s2(k),mdl{k}]=calc_svm(TrainC{i1},TrainC{i2},NTr(i1),NTr(i2));
            [~, dT] = predict(mdl{k},TestC);
            d=dT(:,2);
            p1=normprb(d/m1(k),1,abs(s1(k)/m1(k)));
            p2=normprb(d/m2(k),1,abs(s2(k)/m2(k)));
            AbsProb(:,i1)=AbsProb(:,i1).*p1;
            AbsProb(:,i2)=AbsProb(:,i2).*p2;
            ProbNorm(:,i1)=ProbNorm(:,i1).*p1*(abs(s1(k)/m1(k))*sqrt(2*pi));
            ProbNorm(:,i2)=ProbNorm(:,i2).*p2*(abs(s2(k)/m2(k))*sqrt(2*pi));
        end
        RelProb=AbsProb./repmat(sum(AbsProb,2),1,NC);
        sigma_tilde=sqrt(2/(1-NC)*log(ProbNorm));
        [~,Id]=max(RelProb,[],2);
        lc=false(NT,1);
        for k=1:NT
            lc(k)=sigma_tilde(k,Id(k))>sf;
        end
        svmmdl.m1=m1;
        svmmdl.m2=m2;
        svmmdl.s1=s1;
        svmmdl.s2=s2;
        svmmdl.mdl=mdl;
        Te.lc=lc;
        Te.Id=Id;
        Te.st=sigma_tilde;
    end

    function Im=createImage(Te)
        sat=zeros(NTe,1);
        for k=1:NTe
            sat(k)=1-Te.st(k,Te.Id(k))/max_sigma;
        end
        sat(sat<0)=0;
        
        Im=zeros(l*w,3);
        Im(:,1)=(Te.Id-1)/NClasses;
        Im(:,2)=sat;
        Im(:,3)=1;
        Im=reshape(hsv2rgb(Im),l,w,3);
    end

end
