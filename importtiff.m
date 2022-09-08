function [d,l,w,nCh,nSl,nFr,m]=importtiff(f)
warning('off','all')
err=[];
t = Tiff(f,'r');

l=t.getTag('ImageLength');
w=t.getTag('ImageWidth');
tagstructure=t.getTagNames;

info=imfinfo(f);
nIm1=length(info);

try
    info=t.getTag('ImageDescription');
catch err
end
if isempty(err)
    pIm = strfind(info,'images');
    if isempty(pIm)
        nIm2=1;
    else
        nIm2=sscanf(info,['%*' num2str(pIm+6) 'c %u %*s']);
    end
else
    nIm2=1;
end
nIm=max(nIm1,nIm2);
if isempty(err)
    if ~isempty(strfind(info,'hyperstack=true'));
        pCh = strfind(info,'channels');
        pSl = strfind(info,'slices');
        pFr = strfind(info,'frames');
        if ~isempty(pCh)
            nCh=sscanf(info,['%*' num2str(pCh+8) 'c %u %*s'],1);
        else
            nCh=1;
        end
        if ~isempty(pSl)
            nSl=sscanf(info,['%*' num2str(pSl+6) 'c %u %*s'],1);
        else
            nSl=1;
        end
        if ~isempty(pFr)
            nFr=sscanf(info,['%*' num2str(pFr+6) 'c %u %*s'],1);
        else
            nFr=1;
        end
    else
        nCh=1; nSl=1; nFr=nIm;
    end
else
    nCh=1; nSl=1; nFr=nIm;
end


d=zeros(l,w,nIm,'single');

for i=1:nIm
    t.setDirectory(i);
    d(:,:,i)=single(t.read());
end

try
    t.getTag('Software');
catch err
end
if isempty(err)
    if ~isempty(t.getTag('Software'))
        pMax=strfind(t.getTag('Software'),'Max');
        if ~isempty(pMax)
            dummy=sscanf(t.getTag('Software'),['%*' num2str(pMax+4) 'c %f %*c']);
            m(1)=dummy(1);
        else
            m(1)=1;
        end
        pMin=strfind(t.getTag('Software'),'Min');
        if ~isempty(pMin)
            dummy=sscanf(t.getTag('Software'),['%*' num2str(pMin+4) 'c %f %*c']);
            m(2)=dummy(1);
        else
            m(2)=0;
        end
        pwvn=strfind(t.getTag('Software'),'Wvn Origin');
        if ~isempty(pwvn)
            dummy=sscanf(t.getTag('Software'),['%*' num2str(pwvn+11) 'c %f %*c']);
            m(3)=dummy(1);
        else
            m(3)=0;
        end
    else
        m(1)=1;
        m(2)=0;
        m(3)=0;
    end
else
    m(1)=1;
    m(2)=0;
    m(3)=0;
end
t.close()
warning('on','all')
end