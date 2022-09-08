function SparseSamplingClassification_Example(sf,max_sigma)
%Reference:Low-energy electron microscopy intensity-voltage data --
%factorization, sparse sampling, and classification"
%Francesco Masia, Wolfgang Langbein, Simon Fischer, Jon-Olaf Krisponeit,
%Jens Falta
%https://arxiv.org/abs/2203.12353

%This function show how to use the SparseSamplingClassification.m function,
%which determines the optimal sampling of hyperspectral images based on the
%minimisation of the classification figure of demerit

%sf: threshold value for sigma_tilde identify low confidence classification
%max_sigma: defines the maximum value of sigma used in the visualisation

%Written by Francesco Masia 2022
%Please send bug reports, comments, or questions to Francesco Masia (masiaf@cf.ac.uk).
%This code comes with no guarantee or warranty of any kind.

nfeatures=[9,36,72,144,288];



Test_D=double(readtiff('PrOx_TrainTest.tif'));%LEEM Data for Training and Testing - full spectral information
Test_D=reshape(Test_D,[],size(Test_D,3))';

Val_D=double(readtiff('PrOx_Val.tif'));%LEEM Data for Validation - full spectral information
Val_D=reshape(Val_D,[],size(Val_D,3))';

F_Spectra=dlmread('PrOx_FSC3_S_TrainTest.dat');%FSC3 spectra of Training and testing region
F_Spectra=F_Spectra(:,2:end);



Mask1=~imread('Mask_1.tif');%Masks to define Training ROIs
Mask2=~imread('Mask_2.tif');
Mask3=~imread('Mask_3.tif');
Mask4=~imread('Mask_4.tif');
Mask5=~imread('Mask_5.tif');
Mask6=~imread('Mask_6.tif');
Mask7=~imread('Mask_7.tif');

%%Defining testing pixels using chessboard pattern
l0=size(Mask1,1);
w0=size(Mask1,2);
l4=floor(l0/4);
w4=floor(w0/4);
A=[true(l4,w4);false(2,w4);false(l4,w4)];
A1=cat(1,A,false(2,w4),A);
nA=[false(l4,w4);false(2,w4);true(l4,w4)];
nA1=cat(1,nA,false(2,w4),nA);
B=cat(2,cat(1,A1,false(l0+6-size(A1,1),size(A1,2))),false(l0+6,2),cat(1,nA1,false(l0+6-size(A1,1),size(A1,2))));
oB=cat(2,cat(1,nA1,false(l0+6-size(A1,1),size(A1,2))),false(l0+6,2),cat(1,A1,false(l0+6-size(A1,1),size(A1,2))));
B1=cat(2,B,false(size(B,1),2),B);
oB1=cat(2,oB,false(size(B,1),2),oB);
B2=cat(2,B1,false(size(B1,1),w0+6-size(B1,2)));
oB2=cat(2,oB1,false(size(oB1,1),w0+6-size(oB1,2)));

sA=[true(l4,w4);false(l4,w4)];
sA1=cat(1,sA,sA);
snA=[false(l4,w4);true(l4,w4)];
snA1=cat(1,snA,snA);
sB=cat(2,cat(1,sA1,false(l0-size(sA1,1),size(sA1,2))),cat(1,snA1,false(l0-size(sA1,1),size(sA1,2))));
sB1=cat(2,sB,sB);
sB2=cat(2,sB1,false(size(sB1,1),w0-size(sB1,2)));

fl=floor(l0/4);
fw=floor(w0/4);
B1=B2(1:(4*fl+6),1:(4*fw+6));
oB1=oB2(1:(4*fl+6),1:(4*fw+6));

fl=floor(l0/4);
fw=floor(w0/4);

Mask1=reshape(Mask1,[],1);
Mask1=Mask1(sB2(:),1);
Mask2=reshape(Mask2,[],1);
Mask2=Mask2(sB2(:),1);
Mask3=reshape(Mask3,[],1);
Mask3=Mask3(sB2(:),1);
Mask4=reshape(Mask4,[],1);
Mask4=Mask4(sB2(:),1);
Mask5=reshape(Mask5,[],1);
Mask5=Mask5(sB2(:),1);
Mask6=reshape(Mask6,[],1);
Mask6=Mask6(sB2(:),1);
Mask7=reshape(Mask7,[],1);
Mask7=Mask7(sB2(:),1);

%Project Training and testing Set to FSC3 components
Test_C=nnlsm_blockpivot(F_Spectra,Test_D, 0,rand(size(F_Spectra,2),size(Test_D,2)))';

%Defining Training populations
Train_C=cell(1,7);
Train_C{1}=Test_C(Mask1(:),:);
Train_C{2}=Test_C(Mask2(:),:);
Train_C{3}=Test_C(Mask3(:),:);
Train_C{4}=Test_C(Mask4(:),:);
Train_C{5}=Test_C(Mask5(:),:);
Train_C{6}=Test_C(Mask6(:),:);
Train_C{7}=Test_C(Mask7(:),:);

NClasses=length(Train_C);
NTr=zeros(1,NClasses);
for i=1:NClasses
    NTr(i)=size(Train_C{i},1);
end

NTe=size(Test_C,1);

for i=1:length(sf)
    %Train svm model and return classification of Test data with full
    %spectral information
    [svmmdl,Te.full]=calc_class(NTe,NClasses,Train_C,NTr,Test_C,sf(i));
    
    for j=1:length(nfeatures)
        %Using the trained svm model, return classification of Test and Validation  data
        %with sparse spectral information using an equidistant sampling
        [TeEq,ValEq,ind_h,f_h,ind_min,f_min,T]=SparseSamplingClassification(NClasses,Test_D,Val_D,F_Spectra,svmmdl,Te,0,sf(i),nfeatures(j));
        effnfeatures=length(ind_min);
        dlmwrite(['PrOx_SparseS_Class_Eq_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_index.dat'],ind_min');
        dlmwrite(['PrOx_SparseS_Class_Eq_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_index_h.dat'],ind_h);
        dlmwrite(['PrOx_SparseS_Class_Eq_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_f_h.dat'],f_h);
        writeFOMinfo(['PrOx_SparseS_Class_Eq_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_FOMInfo.dat'],TeEq,ValEq,f_min,T);
        Im=createImage(Te.full,ValEq.SS);
        imwrite(Im,['PrOx_SparseS_Class_Eq_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_maxSigma' num2str(max_sigma) '.tif']);
        
        %Using the trained svm model, return classification of Test and Validation  data
        %with sparse spectral information using a random sampling and
        %minimising the Loss of information
        [TeR,ValR,ind_h,f_h,ind_min,f_min,T]=SparseSamplingClassification(NClasses,Test_D,Val_D,F_Spectra,svmmdl,Te,1,sf(i),nfeatures(j),'minimize','lossinfo');
        effnfeatures=length(ind_min);
        dlmwrite(['PrOx_SparseS_Class_fl_Rand_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_index.dat'],ind_min');
        dlmwrite(['PrOx_SparseS_Class_fl_Rand_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_index_h.dat'],ind_h);
        dlmwrite(['PrOx_SparseS_Class_fl_Rand_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_f_h.dat'],f_h);
        writeFOMinfo(['PrOx_SparseS_Class_fl_Rand_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_FOMInfo.dat'],TeR,ValR,f_min,T);
        Im=createImage(Te.full,ValR.SS);
        imwrite(Im,['PrOx_SparseS_Class_fl_Rand_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_maxSigma' num2str(max_sigma) '.tif']);
        
        %Using the trained svm model, return classification of Test and Validation  data
        %with sparse spectral information using a random sampling and
        %minimising the mis-classification
        [TeR,ValR,ind_h,f_h,ind_min,f_min,T]=SparseSamplingClassification(NClasses,Test_D,Val_D,F_Spectra,svmmdl,Te,1,sf(i),nfeatures(j),'minimize','misclass');
        effnfeatures=length(ind_min);
        dlmwrite(['PrOx_SparseS_Class_fm_Rand_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_index.dat'],ind_min');
        dlmwrite(['PrOx_SparseS_Class_fm_Rand_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_index_h.dat'],ind_h);
        dlmwrite(['PrOx_SparseS_Class_fm_Rand_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_f_h.dat'],f_h);
        writeFOMinfo(['PrOx_SparseS_Class_fm_Rand_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_FOMInfo.dat'],TeR,ValR,f_min,T);
        Im=createImage(Te.full,ValR.SS);
        imwrite(Im,['PrOx_SparseS_Class_fm_Rand_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_maxSigma' num2str(max_sigma) '.tif']);
        
        %Using the trained svm model, return classification of Test and Validation  data
        %with sparse spectral information using a random walk sampling and
        %minimising the Loss of information
        [TeRW,ValRW,ind_h,f_h,ind_min,f_min,T]=SparseSamplingClassification(NClasses,Test_D,Val_D,F_Spectra,svmmdl,Te,2,sf(i),nfeatures(j),'minimize','lossinfo');
        effnfeatures=length(ind_min);
        dlmwrite(['PrOx_SparseS_Class_fl_RandWalk_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_index.dat'],ind_min');
        dlmwrite(['PrOx_SparseS_Class_fl_RandWalk_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_index_h.dat'],ind_h);
        dlmwrite(['PrOx_SparseS_Class_fl_RandWalk_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_f_h.dat'],f_h);
        writeFOMinfo(['PrOx_SparseS_Class_fl_RandWalk_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_FOMInfo.dat'],TeRW,ValRW,f_min,T);
        Im=createImage(Te.full,ValRW.SS);
        imwrite(Im,['PrOx_SparseS_Class_fl_RandWalk_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_maxSigma' num2str(max_sigma) '.tif']);
        
        %Using the trained svm model, return classification of Test and Validation  data
        %with sparse spectral information using a random walk sampling and
        %minimising the mis-classification
        [TeRW,ValRW,ind_h,f_h,ind_min,f_min,T]=SparseSamplingClassification(NClasses,Test_D,Val_D,F_Spectra,svmmdl,Te,2,sf(i),nfeatures(j),'minimize','misclass');
        effnfeatures=length(ind_min);
        dlmwrite(['PrOx_SparseS_Class_fm_RandWalk_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_index.dat'],ind_min');
        dlmwrite(['PrOx_SparseS_Class_fm_RandWalk_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_index_h.dat'],ind_h);
        dlmwrite(['PrOx_SparseS_Class_fm_RandWalk_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_f_h.dat'],f_h);
        writeFOMinfo(['PrOx_SparseS_Class_fm_RandWalk_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_FOMInfo.dat'],TeRW,ValRW,f_min,T);
        Im=createImage(Te.full,ValRW.SS);
        imwrite(Im,['PrOx_SparseS_Class_fm_RandWalk_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_maxSigma' num2str(max_sigma) '.tif']);
    
        %Using the trained svm model, return classification of Test and Validation  data
        %with sparse spectral information using a gradient-based  sampling and
        %minimising the Loss of information
        [TeG,ValG,ind_h,f_h,ind_min,f_min,T]=SparseSamplingClassification(NClasses,Test_D,Val_D,F_Spectra,svmmdl,Te,6,sf(i),nfeatures(j));
        effnfeatures=length(ind_min);
        dlmwrite(['PrOx_SparseS_Class_fl_Grad_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_index.dat'],ind_min');
        writeFOMinfo(['PrOx_SparseS_Class_fl_Grad_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_FOMInfo.dat'],TeG,ValG,f_min,T);
        Im=createImage(Te.full,ValG.SS);
        imwrite(Im,['PrOx_SparseS_Class_fl_Grad_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_maxSigma' num2str(max_sigma) '.tif']);

         %Using the trained svm model, return classification of Test and Validation  data
        %with sparse spectral information using a gradient-based  sampling and
        %minimising the mis-classification
        [TeG,ValG,ind_h,f_h,ind_min,f_min,T]=SparseSamplingClassification(NClasses,Test_D,Val_D,F_Spectra,svmmdl,Te,6,sf(i),nfeatures(j),'minimize','misclass');
        effnfeatures=length(ind_min);
        dlmwrite(['PrOx_SparseS_Class_fm_Grad_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_index.dat'],ind_min');
        writeFOMinfo(['PrOx_SparseS_Class_fm_Grad_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_FOMInfo.dat'],TeG,ValG,f_min,T);
        Im=createImage(Te.full,ValG.SS);
        imwrite(Im,['PrOx_SparseS_Class_fm_Grad_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_maxSigma' num2str(max_sigma) '.tif']);
       
        %Using the trained svm model, return classification of Test and Validation  data
        %with sparse spectral information using a surrogate-based  sampling and
        %minimising the Loss of information
        [TeSur,ValSur,ind_h,f_h,ind_min,f_min,T]=SparseSamplingClassification(NClasses,Test_D,Val_D,F_Spectra,svmmdl,Te,5,sf(i),nfeatures(j));
        effnfeatures=length(ind_min);
        dlmwrite(['PrOx_SparseS_Class_fl_Surr_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_index.dat'],ind_min');
        dlmwrite(['PrOx_SparseS_Class_fl_Surr_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_index_h.dat'],ind_h');
        dlmwrite(['PrOx_SparseS_Class_fl_Surr_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_f_h.dat'],f_h');
        writeFOMinfo(['PrOx_SparseS_Class_fl_Surr_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_FOMInfo.dat'],TeSur,ValSur,f_min,T);
        Im=createImage(Te.full,ValSur.SS);
        imwrite(Im,['PrOx_SparseS_Class_fl_Surr_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_maxSigma' num2str(max_sigma) '.tif']);
        
        %Using the trained svm model, return classification of Test and Validation  data
        %with sparse spectral information using a surrogate-based  sampling and
        %minimising the mis-classification
        [TeSur,ValSur,ind_h,f_h,ind_min,f_min,T]=SparseSamplingClassification(NClasses,Test_D,Val_D,F_Spectra,svmmdl,Te,5,sf(i),nfeatures(j),'minimize','misclass');
        effnfeatures=length(ind_min);
        dlmwrite(['PrOx_SparseS_Class_fm_Surr_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_index_h.dat'],ind_h');
        dlmwrite(['PrOx_SparseS_Class_fm_Surr_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_f_h.dat'],f_h');
        dlmwrite(['PrOx_SparseS_Class_fm_Surr_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_index.dat'],ind_min');
        writeFOMinfo(['PrOx_SparseS_Class_fm_Surr_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_FOMInfo.dat'],TeSur,ValSur,f_min,T);
        Im=createImage(Te.full,ValSur.SS);
        imwrite(Im,['PrOx_SparseS_Class_fm_Surr_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_maxSigma' num2str(max_sigma) '.tif']);

    end
    
    %Return classification of Test and Validation data
    %with sparse spectral information using a forward feature selection method and
    %minimising the loss of information
    [TeWr,ValWr,ind_h,f_h,ind_min,f_min,T]=SparseSamplingClassification(NClasses,Test_D,Val_D,F_Spectra,svmmdl,Te,3,sf(i),nfeatures(1),'minimize','lossinfo','direction','forward');
    save(['PrOx_SparseS_Class_fl_SeqF_sf' num2str(sf(i)) '_N' num2str(nfeatures(1)) '.mat'],'TeWr','ValWr','ind_h','f_h','ind_min','f_min','T');
    dlmwrite(['PrOx_SparseS_Class_fl_SeqF_sf' num2str(sf(i)) '_N' num2str(nfeatures(1)) '_index.dat'],ind_min');
    dlmwrite(['PrOx_SparseS_Class_fl_SeqF_sf' num2str(sf(i)) '_N' num2str(nfeatures(1)) '_index_h.dat'],ind_h);
    dlmwrite(['PrOx_SparseS_Class_fl_SeqF_sf' num2str(sf(i)) '_N' num2str(nfeatures(1)) '_f_h.dat'],f_h);
    ind_W=ind_min;
    CTr=nnlsm_blockpivot(F_Spectra(ind_W,:),Test_D(ind_W,:), 0,rand(size(F_Spectra,2),size(Test_D,2)))';
    [TeW.SS.Id,TeW.SS.lc,TeW.SS.st]=proj_class(NClasses,CTr,svmmdl);
    CTr=nnlsm_blockpivot(F_Spectra(ind_W,:),Val_D(ind_W,:), 0,rand(size(F_Spectra,2),size(Test_D,2)))';
    [ValW.SS.Id,ValW.SS.lc,ValW.SS.st]=proj_class(NClasses,CTr,svmmdl);
    ValW.full=Val.full;
    effnfeatures=length(ind_W);
    writeFOMinfo(['PrOx_SparseS_Class_fl_SeqF_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_FOMInfo.dat'],TeW,ValW,f_min,T);
    Im=createImage(Te.full,ValW.SS);
    imwrite(Im,['PrOx_SparseS_Class_fl_SeqF_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_maxSigma' num2str(max_sigma) '.tif']);
   
    %Return classification of Test and Validation data
    %with sparse spectral information using a forward feature selection method and
    %minimising the mis-classification
    [TeWr,ValWr,ind_h,f_h,ind_min,f_min,T]=SparseSamplingClassification(NClasses,Test_D,Val_D,F_Spectra,svmmdl,Te,3,sf(i),nfeatures(1),'minimize','misclass','direction','forward');
    save(['PrOx_SparseS_Class_fm_SeqF_sf' num2str(sf(i)) '_N' num2str(nfeatures(1)) '.mat'],'TeWr','ValWr','ind_h','f_h','ind_min','f_min','T');
    dlmwrite(['PrOx_SparseS_Class_fm_SeqF_sf' num2str(sf(i)) '_N' num2str(nfeatures(1)) '_index.dat'],ind_min');
    dlmwrite(['PrOx_SparseS_Class_fm_SeqF_sf' num2str(sf(i)) '_N' num2str(nfeatures(1)) '_index_h.dat'],ind_h);
    dlmwrite(['PrOx_SparseS_Class_fm_SeqF_sf' num2str(sf(i)) '_N' num2str(nfeatures(1)) '_f_h.dat'],f_h);
    ind_W=ind_min;
    CTr=nnlsm_blockpivot(F_Spectra(ind_W,:),Test_D(ind_W,:), 0,rand(size(F_Spectra,2),size(Test_D,2)))';
    [TeW.SS.Id,TeW.SS.lc,TeW.SS.st]=proj_class(NClasses,CTr,svmmdl);
    CTr=nnlsm_blockpivot(F_Spectra(ind_W,:),Val_D(ind_W,:), 0,rand(size(F_Spectra,2),size(Test_D,2)))';
    [ValW.SS.Id,ValW.SS.lc,ValW.SS.st]=proj_class(NClasses,CTr,svmmdl);
    ValW.full=Val.full;
    effnfeatures=length(ind_W);
    writeFOMinfo(['PrOx_SparseS_Class_fm_SeqF_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_FOMInfo.dat'],TeW,ValW,f_min,T);
    Im=createImage(Te.full,ValW.SS);
    imwrite(Im,['PrOx_SparseS_Class_fm_SeqF_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_maxSigma' num2str(max_sigma) '.tif']);

    %Return classification of Test and Validation data
    %with sparse spectral information using a backward feature selection method and
    %minimising the loss of information

    [TeWr,ValWr,ind_h,f_h,ind_min,f_min,T]=SparseSamplingClassification(NClasses,Test_D,Val_D,F_Spectra,svmmdl,Te,3,sf(i),nfeatures(1),'minimize','lossinfo');
    save(['PrOx_SparseS_Class_fl_SeqB_sf' num2str(sf(i)) '_N' num2str(nfeatures(1)) '.mat'],'TeWr','ValWr','ind_h','f_h','ind_min','f_min','T');
    dlmwrite(['PrOx_SparseS_Class_fl_SeqB_sf' num2str(sf(i)) '_N' num2str(nfeatures(1)) '_index.dat'],ind_min');
    dlmwrite(['PrOx_SparseS_Class_fl_SeqB_sf' num2str(sf(i)) '_N' num2str(nfeatures(1)) '_index_h.dat'],ind_h);
    dlmwrite(['PrOx_SparseS_Class_fl_SeqB_sf' num2str(sf(i)) '_N' num2str(nfeatures(1)) '_f_h.dat'],f_h);
    
    for i0=length(nfeatures):-1:1
        hcounter=size(F_Spectra,1)-nfeatures(i0)+1;
        ind_W=ind_h(hcounter,:);
        CTr=nnlsm_blockpivot(F_Spectra(ind_W,:),Test_D(ind_W,:), 0,rand(size(F_Spectra,2),size(Test_D,2)))';
        [TeW.SS.Id,TeW.SS.lc,TeW.SS.st]=proj_class(NClasses,CTr,svmmdl);
        CTr=nnlsm_blockpivot(F_Spectra(ind_W,:),Val_D(ind_W,:), 0,rand(size(F_Spectra,2),size(Test_D,2)))';
        [ValW.SS.Id,ValW.SS.lc,ValW.SS.st]=proj_class(NClasses,CTr,svmmdl);
        ValW.full=Val.full;
        effnfeatures=length(ind_W);
        writeFOMinfo(['PrOx_SparseS_Class_fl_SeqB_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_FOMInfo.dat'],TeW,ValW,f_h(hcounter),T);
        Im=createImage(Te.full,ValW.SS);
        imwrite(Im,['PrOx_SparseS_Class_fl_SeqB_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_maxSigma' num2str(max_sigma) '.tif']);
    end

    %Return classification of Test and Validation data
    %with sparse spectral information using a backward feature selection method and
    %minimising the mis-classification
    [TeWr,ValWr,ind_h,f_h,ind_min,f_min,T]=SparseSamplingClassification(NClasses,Test_D,Val_D,F_Spectra,svmmdl,Te,3,sf(i),nfeatures(1),'minimize','misclass');
    save(['PrOx_SparseS_Class_fm_SeqB_sf' num2str(sf(i)) '_N' num2str(nfeatures(1)) '.mat'],'TeWr','ValWr','ind_h','f_h','ind_min','f_min','T');
    dlmwrite(['PrOx_SparseS_Class_fm_SeqB_sf' num2str(sf(i)) '_N' num2str(nfeatures(1)) '_index.dat'],ind_min');
    dlmwrite(['PrOx_SparseS_Class_fm_SeqB_sf' num2str(sf(i)) '_N' num2str(nfeatures(1)) '_index_h.dat'],ind_h);
    dlmwrite(['PrOx_SparseS_Class_fm_SeqB_sf' num2str(sf(i)) '_N' num2str(nfeatures(1)) '_f_h.dat'],f_h);
    
    for i0=length(nfeatures):-1:1
        hcounter=size(F_Spectra,1)-nfeatures(i0)+1;
        ind_W=ind_h(hcounter,:);
        CTr=nnlsm_blockpivot(F_Spectra(ind_W,:),Test_D(ind_W,:), 0,rand(size(F_Spectra,2),size(Test_D,2)))';
        [TeW.SS.Id,TeW.SS.lc,TeW.SS.st]=proj_class(NClasses,CTr,svmmdl);
        CTr=nnlsm_blockpivot(F_Spectra(ind_W,:),Val_D(ind_W,:), 0,rand(size(F_Spectra,2),size(Test_D,2)))';
        [ValW.SS.Id,ValW.SS.lc,ValW.SS.st]=proj_class(NClasses,CTr,svmmdl);
        ValW.full=Val.full;
        effnfeatures=length(ind_W);
        writeFOMinfo(['PrOx_SparseS_Class_fm_SeqB_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_FOMInfo.dat'],TeW,ValW,f_h(hcounter),T);
        Im=createImage(Te.full,ValW.SS);
        imwrite(Im,['PrOx_SparseS_Class_fm_SeqB_sf' num2str(sf(i)) '_N' num2str(effnfeatures) '_maxSigma' num2str(max_sigma) '.tif']);
    end
end

    function [Id,lc,sigma_tilde]=proj_class(NC,C,svmmdl)
        NT=size(C,1);
        combs=nchoosek(1:NC,2);
        AbsProb=ones(NT,NC);
        ProbNorm=ones(NT,NC);
        for k=1:size(combs,1)
            i1=combs(k,1);
            i2=combs(k,2);
            [~, dT] = predict(svmmdl.mdl{k},C);
            d=dT(:,2);
            p1=normprb(d/svmmdl.m1(k),1,abs(svmmdl.s1(k)/svmmdl.m1(k)));
            p2=normprb(d/svmmdl.m2(k),1,abs(svmmdl.s2(k)/svmmdl.m2(k)));
            AbsProb(:,i1)=AbsProb(:,i1).*p1;
            AbsProb(:,i2)=AbsProb(:,i2).*p2;
            ProbNorm(:,i1)=ProbNorm(:,i1).*p1*(abs(svmmdl.s1(k)/svmmdl.m1(k))*sqrt(2*pi));
            ProbNorm(:,i2)=ProbNorm(:,i2).*p2*(abs(svmmdl.s2(k)/svmmdl.m2(k))*sqrt(2*pi));
        end
        RelProb=AbsProb./repmat(sum(AbsProb,2),1,NC);
        sigma_tilde=sqrt(2/(1-NC)*log(ProbNorm));
        [~,Id]=max(RelProb,[],2);
        lc=false(NT,1);
        for k=1:NT
            lc(k)=sigma_tilde(k,Id(k))>sf;
        end
    end
    function [svmmdl,Te]=calc_class(NT,NC,TrainC,NTr,TestC,sf)
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
    function Im=createImage(Te,Val)
        sat=zeros(NTe,1);
        for k=1:NTe
            sat(k)=1-Te.st(k,Te.Id(k))/max_sigma;
        end
        sat(sat<0)=0;
        
        satVal=zeros(NTe,1);
        for k=1:NTe
            satVal(k)=1-Val.st(k,Val.Id(k))/max_sigma;
        end
        satVal(satVal<0)=0;
        
        Im=zeros((fl*4+6)*(fw*4+6),3);
        Im(B1(:),1)=(Te.Id-1)/NClasses;
        Im(oB1(:),1)=(Val.Id-1)/NClasses;
        Im(B1(:),2)=sat;
        Im(oB1(:),2)=satVal;
        Im(B1(:),3)=1;
        Im(oB1(:),3)=1;
        Im=reshape(hsv2rgb(Im),fl*4+6,fw*4+6,3);
    end
    function writeFOMinfo(fname,Te,Val,f_m,T)
        f_li_Te=sum(Te.SS.lc(~Te.full.lc));
        f_mc_Te(3)=sum(Te.SS.Id(~Te.full.lc)~=Te.full.Id(~Te.full.lc));
        f_mc_Te(2)=sum(Te.SS.Id(~Te.SS.lc)~=Te.full.Id(~Te.SS.lc));
        f_mc_Te(1)=sum(Te.SS.Id~=Te.full.Id);
        
        f_li_Val=sum(Val.SS.lc(~Val.full.lc));
        f_mc_Val(3)=sum(Val.SS.Id(~Val.full.lc)~=Val.full.Id(~Val.full.lc));
        f_mc_Val(2)=sum(Val.SS.Id(~Val.SS.lc)~=Val.full.Id(~Val.SS.lc));
        f_mc_Val(1)=sum(Val.SS.Id~=Val.full.Id);
        
        fid=fopen(fname,'w');
        fprintf(fid,'Number of features %d\n',effnfeatures);
        fprintf(fid,'Elapsed Time %f\n',T);
        fprintf(fid,'Min FOM %f\n',f_m);
        fprintf(fid,'Test ROI\n');
        fprintf(fid,'Loss Information %f\n',f_li_Te/size(Test_D,2));
        fprintf(fid,'MisClassification (1) %f\n',f_mc_Te(1)/size(Test_D,2));
        fprintf(fid,'MisClassification (2) %f\n',f_mc_Te(2)/size(Test_D,2));
        fprintf(fid,'MisClassification (3) %f\n',f_mc_Te(3)/size(Test_D,2));
        fprintf(fid,'Validation ROI\n');
        fprintf(fid,'Loss Information %f\n',f_li_Val/size(Test_D,2));
        fprintf(fid,'MisClassification (1) %f\n',f_mc_Val(1)/size(Test_D,2));
        fprintf(fid,'MisClassification (2) %f\n',f_mc_Val(2)/size(Test_D,2));
        fprintf(fid,'MisClassification (3) %f\n',f_mc_Val(3)/size(Test_D,2));
        fclose(fid);
    end

end
