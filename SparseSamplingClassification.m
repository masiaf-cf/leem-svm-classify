function [Te,Val,ind_h,f_h,ind_min,f_min,T]=SparseSamplingClassification(NClasses,Test_D,Val_D,F_Spectra,svmmdl,Te,method,sf,nfeatures,varargin)
%Reference:Low-energy electron microscopy intensity-voltage data --
%factorization, sparse sampling, and classification"
%Francesco Masia, Wolfgang Langbein, Simon Fischer, Jon-Olaf Krisponeit,
%Jens Falta
%https://arxiv.org/abs/2203.12353

%Input Variables
%NClasses: define Number of classes

%Test_D: p x s array containing the Hyperspectral Data used to test the
%classification. p is number of spatial points, s is the number of
%spectral points

%Val_D: p x s array containing the Hyperspectral Data used to validate the
%classification

%F-Spectra: 1 x s array containing the spectra obtained by the FSC3
%analysis to project the data in the feature subspace

%svmmdl: classification model

%Te: structure containing the results of the classification on the test set
%using the full spectral information, (Te.full.Id: 1 x p array containing
%the class identifier, Te.full.st: 1 x p array containing the value of
%sigma_tilde as defined in the paper, Te.full.lc: 1 x p logical array
%showing if the sigma_tilde is above the threshold sf)

%method: sparse sampling method. 0: equidistant points. 1: Random points.
%2: Random walk. 3: deterministic feature selection

%sf: threshold value for sigma_tilde identify low confidence classification

%nfeatures: define the number of feature (spectral points) to consider for
%the sampling. For deterministic feature selection, it gives the minimum
%(maximum) number of features for a backward (forward) method

%Optional input

%'minimize' defines what figure of merit to minimize. Accepted values:
%'lossinfo' or 'misclass'. Refer to the publication for definition

%'direction' defines the direction for the deterministic feature selection 
%method. Accepted values: 'backward' or 'forward'.

%Output Variables
%Te: structure containing the results of the classification on the test set
%using the full and sparse sampled spectral information

%Val: structure containing the results of the classification on the validation set
%using the full and sparse sampled spectral information

%ind_h: array showing the evolution of the spectral points indeces vs the
%iteration step

%f_h: array showing the evolution of the value of the figure of merit vs
%the iteration step

%ind_min: final set of spectral indeces which minimises the fom

%f_min: value of the fom at ind_min

%T: computation time

%Written by Francesco Masia 2022
%Please send bug reports, comments, or questions to Francesco Masia (masiaf@cf.ac.uk).
%This code comes with no guarantee or warranty of any kind.

defaultIter = 1000;
defaultdirection='backward';
defaultMinimization='lossinfo';
defaultFTh=0.001;

expectedMinimization={'lossinfo','misclass'};
expecteddirection={'forward','backward'};

p = inputParser;
validScalarPosNum = @(x) isnumeric(x) && isscalar(x) && (x > 0);

addRequired(p,'NClasses',validScalarPosNum);
addRequired(p,'Test_D',@(x) isnumeric(x));
addRequired(p,'Val_D',@(x) isnumeric(x));
addRequired(p,'F_Spectra',@(x) isnumeric(x));
addRequired(p,'svmmdl',@(x) isstruct(x));
addRequired(p,'Te',@(x) isstruct(x));
addRequired(p,'method',@(x) isnumeric(x) && isscalar(x));
addRequired(p,'sf',@(x) isnumeric(x));
addRequired(p,'nfeatures',validScalarPosNum);

addParameter(p,'minimize',defaultMinimization,@(x) any(validatestring(x,expectedMinimization)));
addParameter(p,'direction',defaultdirection,@(x) any(validatestring(x,expecteddirection)));
addParameter(p,'NIter',defaultIter,validScalarPosNum);
addParameter(p,'Threshold',defaultFTh,validScalarPosNum);

parse(p,NClasses,Test_D,Val_D,F_Spectra,svmmdl,Te,method,sf,nfeatures,varargin{:});
minimize = p.Results.minimize;
NIter = p.Results.NIter;
direction=p.Results.direction;
f_th=p.Results.Threshold;


Val_D_full=nnlsm_blockpivot(F_Spectra,Val_D, 0,rand(size(F_Spectra,2),size(Val_D,2)))';
[Val.full.Id,Val.full.lc,Val.full.st]=proj_class(NClasses,Val_D_full,svmmdl,sf);
nsp=size(F_Spectra,1);
sp=1:nsp;
tic
switch method
    case 0
        stepi=floor(nsp/nfeatures+1);
        start=1+floor((nsp-max(1:stepi:nsp))/2);
        ind_min=start:stepi:nsp;
        Test_C_SS=nnlsm_blockpivot(F_Spectra(ind_min,:),Test_D(ind_min,:), 0,rand(size(F_Spectra,2),size(Test_D,2)))';
        [Te.SS.Id,Te.SS.lc,Te.SS.st]=proj_class(NClasses,Test_C_SS,svmmdl,sf);
        if strcmp(minimize,'lossinfo')
            f_min=sum(Te.SS.lc(~Te.full.lc));
        else
            f_min=sum(Te.SS.Id~=Te.full.Id);
        end
        ind_h=ind_min;
        f_h=f_min;
    case 1
        f_h=zeros(NIter,1);
        ind_h=zeros(NIter,nfeatures);
        fulllc=Te.full.lc;
        fullId=Te.full.Id;
        lcTe=zeros(length(fulllc),NIter);
        IdTe=zeros(length(fulllc),NIter);
        stTe=zeros(length(fulllc),NClasses,NIter);
        parfor r_c=1:NIter
            ind_h(r_c,:)=sort(randsample(1:nsp,nfeatures),'ascend');
            Test_C_SS=nnlsm_blockpivot(F_Spectra(ind_h(r_c,:),:),Test_D(ind_h(r_c,:),:), 0,rand(size(F_Spectra,2),size(Test_D,2)))';
            [IdTe(:,r_c),lcTe(:,r_c),stTe(:,:,r_c)]=proj_class(NClasses,Test_C_SS,svmmdl,sf);
            if strcmp(minimize,'lossinfo')
                f_h(r_c)=sum(lcTe(:,r_c).*~fulllc);
            else
                f_h(r_c)=sum(IdTe(:,r_c)~=fullId);
            end
        end
        [f_min,it_min]=min(f_h);
        ind_min=ind_h(it_min,:);
        Te.SS.Id=IdTe(:,it_min);
        Te.SS.lc=lcTe(:,it_min);
        Te.SS.st=stTe(:,:,it_min);
    case 2
        stepi=floor(nsp/nfeatures+1);
        start=1+floor((nsp-max(1:stepi:nsp))/2);
        ind_eq=start:stepi:nsp;
        Test_C_SS=nnlsm_blockpivot(F_Spectra(ind_eq,:),Test_D(ind_eq,:), 0,rand(size(F_Spectra,2),size(Test_D,2)))';
        [IdTe,lcTe,stTe]=proj_class(NClasses,Test_C_SS,svmmdl,sf);
        ok=0;
        f_h=zeros(NIter+1,1);
        ind_h=zeros(NIter+1,length(ind_eq));
        if strcmp(minimize,'lossinfo')
            f_h(1)=sum(lcTe(~Te.full.lc));
        else
            f_h(1)=sum(IdTe~=Te.full.Id);
        end
        f_min=f_h(1);
        ind_h(1,:)=ind_eq;
        ind_min=ind_eq;
        Te.SS.Id=IdTe;
        Te.SS.lc=lcTe;
        Te.SS.st=stTe;
        r_c=0;
        hwb=waitbar(0);
        while ~ok
            r_c=r_c+1;
            extind=[1,ind_min,nsp];
            delta=diff(extind)*0.5;
            deltarand=zeros(1,length(ind_eq));
            for jj=1:length(ind_eq)
                deltarand(jj)=(delta(jj)+delta(jj+1))*rand(1)-delta(jj);
            end
            ind_h(r_c+1,:)=ind_min+round(deltarand);
            ind_h(r_c+1,1)=max(ind_h(r_c+1,1),1);
            ind_h(r_c+1,end)=min(ind_h(r_c+1,end),nsp);
            Test_C_SS=nnlsm_blockpivot(F_Spectra(ind_h(r_c+1,:),:),Test_D(ind_h(r_c+1,:),:), 0,rand(size(F_Spectra,2),size(Test_D,2)))';
            [IdTe,lcTe,stTe]=proj_class(NClasses,Test_C_SS,svmmdl,sf);
            if strcmp(minimize,'lossinfo')
                f_h(r_c+1)=sum(lcTe(~Te.full.lc));
            else
                f_h(r_c+1)=sum(IdTe~=Te.full.Id);
            end
            if r_c>=NIter || f_h(r_c+1)/(size(Test_D,2))<f_th
                ok=1;
            end
            if f_h(r_c+1)<f_min
                Te.SS.Id=IdTe;
                Te.SS.lc=lcTe;
                Te.SS.st=stTe;
                ind_min=ind_h(r_c+1,:);
                f_min=f_h(r_c+1);
            end
            waitbar(r_c/NIter,hwb);
        end
        delete(hwb)
    case 3
        opts=statset('UseParallel',true);
        [inmodel,history] = sequentialfs(@funW,Test_D',Te.full.Id,'cv','none','nfeatures',nfeatures,'direction',direction,'options',opts);
        f_h=history.Crit;
        ind_h=history.In;
        ind_min=sp(inmodel);
        f_min=f_h(end);
        Test_C_SS=nnlsm_blockpivot(F_Spectra(ind_min,:),Test_D(ind_min,:), 0,rand(size(F_Spectra,2),size(Test_D,2)))';
        [Te.SS.Id,Te.SS.lc,Te.SS.st]=proj_class(NClasses,Test_C_SS,svmmdl,sf);
    case 4
        opts=statset('UseParallel',true);
        [inmodel,history] = sequentialfs(@funW,Test_D',Te.full.Id,'cv','none','nfeatures',nfeatures,'direction',direction,'options',opts);
        f_h=history.Crit;
        ind_h=history.In;
        ind_min=sp(inmodel);
        f_min=f_h(end);
        Test_C_SS=nnlsm_blockpivot(F_Spectra(ind_min,:),Test_D(ind_min,:), 0,rand(size(F_Spectra,2),size(Test_D,2)))';
        [Te.SS.Id,Te.SS.lc,Te.SS.st]=proj_class(NClasses,Test_C_SS,svmmdl,sf);
    case 5
        stepi=floor(nsp/nfeatures+1);
        start=1+floor((nsp-max(1:stepi:nsp))/2);
        ind0=start:stepi:nsp;
        options = optimoptions('surrogateopt','InitialPoints',ind0,'Display','off','MaxFunctionEvaluations',1000,'UseParallel',true);
        [indO,f_min,~,~,trials] = surrogateopt(@minfun,ones(nfeatures,1),nsp*ones(nfeatures,1),1:nfeatures,options);
        ind_min=round(sort(indO));
        ind_h=trials.X;
        f_h=trials.Fval;
        Test_C_SS=nnlsm_blockpivot(F_Spectra(ind_min,:),Test_D(ind_min,:), 0,rand(size(F_Spectra,2),size(Test_D,2)))';
        [Te.SS.Id,Te.SS.lc,Te.SS.st]=proj_class(NClasses,Test_C_SS,svmmdl,sf);
    case 6
        stepi=floor(nsp/nfeatures+1);
        start=1+floor((nsp-max(1:stepi:nsp))/2);
        ind0=start:stepi:nsp;
        options = optimoptions('fmincon','DiffMinChange',1) ;
        indO = fmincon(@minfun_int,ind0,[],[],[],[],ones(nfeatures,1),nsp*ones(nfeatures,1),[],options);
        ind_min=round(sort(indO));
        f_min=minfun_int(indO);
        ind_h=ind_min;
        f_h=f_min;
        Test_C_SS=nnlsm_blockpivot(F_Spectra(ind_min,:),Test_D(ind_min,:), 0,rand(size(F_Spectra,2),size(Test_D,2)))';
        [Te.SS.Id,Te.SS.lc,Te.SS.st]=proj_class(NClasses,Test_C_SS,svmmdl,sf);
end
Val_C_SS=nnlsm_blockpivot(F_Spectra(ind_min,:),Val_D(ind_min,:), 0,rand(size(F_Spectra,2),size(Val_D,2)))';
[Val.SS.Id,Val.SS.lc,Val.SS.st]=proj_class(NClasses,Val_C_SS,svmmdl,sf);
T=toc;

    function f=funW(XT,yT,Xt,yt)
        %Defines the figure of demerit to be minimised for the deterministic
        %feature selection method
        %See documentation of sequentialfs
        
        [~,ind_W]=ismember(XT',Test_D,'rows');
        f=minfun(ind_W);
    end

    function f=minfun(X)
        %Calculates the figure of demerit given the spectral points X
        %Input:
        %X: spectral points
        %Output
        %f: figure of demerit
        
        X=sort(X);
        CTr=nnlsm_blockpivot(F_Spectra(X,:),Test_D(X,:), 0,rand(size(F_Spectra,2),size(Test_D,2)))';
        [Id,lc]=proj_class(NClasses,CTr,svmmdl,sf);
        if strcmp(minimize,'lossinfo')
            f=sum(lc(~Te.full.lc));
        else
            f=sum(Id~=Te.full.Id);
        end
    end
    function f=minfun_int(X)
        %Calculates the figure of demerit given the spectral points X and
        %interpolating the data
        %Input:
        %X: spectral points
        %Output
        %f: figure of demerit
        X=sort(X);
        F=interp1(1:nsp,F_Spectra,X);
        D=interp1(1:nsp,Test_D,X);
        CTr=nnlsm_blockpivot(F,D, 0,rand(size(F_Spectra,2),size(Test_D,2)))';
        [Id,lc]=proj_class(NClasses,CTr,svmmdl,sf);
        if strcmp(minimize,'lossinfo')
            f=sum(lc(~Te.full.lc));
        else
            f=sum(Id~=Te.full.Id);
        end
    end
end