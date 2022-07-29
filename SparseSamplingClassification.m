function [Te,Val,ind_h,f_h,ind_min,f_min,T]=SparseSamplingClassification(NClasses,Test_D,Val_D,F_Spectra,svmmdl,Te,method,sf,nfeatures,varargin)
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
[Val.full.Id,Val.full.wa,Val.full.st]=proj_class(NClasses,Val_D_full,svmmdl);
nsp=size(F_Spectra,1);
sp=1:nsp;
tic
switch method
    case 0
        stepi=floor(nsp/nfeatures+1);
        start=1+floor((nsp-max(1:stepi:nsp))/2);
        ind_min=start:stepi:nsp;
        Test_C_SS=nnlsm_blockpivot(F_Spectra(ind_min,:),Test_D(ind_min,:), 0,rand(size(F_Spectra,2),size(Test_D,2)))';
        [Te.SS.Id,Te.SS.wa,Te.SS.st]=proj_class(NClasses,Test_C_SS,svmmdl);
        if strcmp(minimize,'lossinfo')
            f_min=sum(Te.SS.wa(~Te.full.wa));
        else
            f_min=sum(Te.SS.Id~=Te.full.Id);
        end
        ind_h=ind_min;
        f_h=f_min;
    case 1
        ok=0;
        f_h=zeros(NIter,1);
        ind_h=zeros(NIter,nfeatures);
        f_min=Inf;
        r_c=0;
        hwb=waitbar(0);
        while ~ok
            r_c=r_c+1;
            ind_h(r_c,:)=sort(randsample(1:nsp,nfeatures),'ascend');
            Test_C_SS=nnlsm_blockpivot(F_Spectra(ind_h(r_c,:),:),Test_D(ind_h(r_c,:),:), 0,rand(size(F_Spectra,2),size(Test_D,2)))';
            [IdTe,waTe,stTe]=proj_class(NClasses,Test_C_SS,svmmdl);
            if strcmp(minimize,'lossinfo')
                f_h(r_c)=sum(waTe(~Te.full.wa));
            else
                f_h(r_c)=sum(IdTe~=Te.full.Id);
            end
            if r_c>=NIter || f_h(r_c)/(size(Test_D,2))<f_th
                ok=1;
            end
            if f_h(r_c)<f_min
                Te.SS.Id=IdTe;
                Te.SS.wa=waTe;
                Te.SS.st=stTe;
                ind_min=ind_h(r_c,:);
                f_min=f_h(r_c);
            end
            waitbar(r_c/NIter,hwb);
        end
    case 2
        stepi=floor(nsp/nfeatures+1);
        start=1+floor((nsp-max(1:stepi:nsp))/2);
        ind_eq=start:stepi:nsp;
        Test_C_SS=nnlsm_blockpivot(F_Spectra(ind_eq,:),Test_D(ind_eq,:), 0,rand(size(F_Spectra,2),size(Test_D,2)))';
        [IdTe,waTe,stTe]=proj_class(NClasses,Test_C_SS,svmmdl);
        ok=0;
        f_h=zeros(NIter+1,1);
        ind_h=zeros(NIter+1,length(ind_eq));
        if strcmp(minimize,'lossinfo')
            f_h(1)=sum(waTe(~Te.full.wa));
        else
            f_h(1)=sum(IdTe~=Te.full.Id);
        end
        f_min=f_h(1);
        ind_h(1,:)=ind_eq;
        ind_min=ind_eq;
        Te.SS.Id=IdTe;
        Te.SS.wa=waTe;
        Te.SS.st=stTe;
        r_c=0;
        hwb=waitbar(0);
        while ~ok
            r_c=r_c+1;
            extind=[1,ind_min,sp];
            delta=diff(extind)*0.5;
            deltarand=zeros(1,length(ind_eq));
            for jj=1:length(ind_eq)
                deltarand(jj)=(delta(jj)+delta(jj+1))*rand(1)-delta(jj);
            end
            ind_h(r_c+1,:)=ind_min+round(deltarand);
            ind_h(r_c+1,1)=max(ind_h(r_c+1,1),1);
            ind_h(r_c+1,end)=min(ind_h(r_c+1,end),nsp);
            Test_C_SS=nnlsm_blockpivot(F_Spectra(ind_h(r_c+1,:),:),Test_D(ind_h(r_c+1,:),:), 0,rand(size(F_Spectra,2),size(Test_D,2)))';
            [IdTe,waTe,stTe]=proj_class(NClasses,Test_C_SS,svmmdl);
            if strcmp(minimize,'lossinfo')
                f_h(r_c+1)=sum(waTe(~Te.full.wa));
            else
                f_h(r_c+1)=sum(IdTe~=Te.full.Id);
            end
            if r_c>=NIter || f_h(r_c+1)/(size(Test_D,2))<f_th
                ok=1;
            end
            if f_h(r_c+1)<f_min
                Te.SS.Id=IdTe;
                Te.SS.wa=waTe;
                Te.SS.st=stTe;
                ind_min=ind_h(r_c+1,:);
                f_min=f_h(r_c+1);
            end
            waitbar(r_c/NIter,hwb);
        end
    case 3
        [inmodel,history] = sequentialfs(@funW,Test_D',Te.full.Id,'cv','none','nfeatures',nfeatures,'direction',direction);
        f_h=history.Crit;
        ind_h=history.In;
        ind_min=sp(inmodel);
        f_min=f_h(end);
        Test_C_SS=nnlsm_blockpivot(F_Spectra(ind_min,:),Test_D(ind_min,:), 0,rand(size(F_Spectra,2),size(Test_D,2)))';
        [Te.SS.Id,Te.SS.wa,Te.SS.st]=proj_class(NClasses,Test_C_SS,svmmdl);
end
Val_C_SS=nnlsm_blockpivot(F_Spectra(ind_min,:),Val_D(ind_min,:), 0,rand(size(F_Spectra,2),size(Val_D,2)))';
[Val.SS.Id,Val.SS.wa,Val.SS.st]=proj_class(NClasses,Val_C_SS,svmmdl);
T=toc;

    function f=funW(XT,yT,Xt,yt)
        [~,ind_W]=ismember(XT',Test_D,'rows');
        CTr=nnlsm_blockpivot(F_Spectra(ind_W,:),Test_D(ind_W,:), 0,rand(size(F_Spectra,2),size(Test_D,2)))';
        [IdTe_W,waTe_W]=proj_class(NClasses,CTr,svmmdl);
        if strcmp(minimize,'lossinfo')
            f=sum(waTe_W(~Te.full.wa));
        else
            f=sum(IdTe_W~=Te.full.Id);
        end
    end

    function [Id,wa,sigma_tilde]=proj_class(NC,C,svmmdl)
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
        wa=false(NT,1);
        for k=1:NT
            wa(k)=sigma_tilde(k,Id(k))>sf;
        end
    end


end