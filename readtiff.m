function I=readtiff(fname)
% fname is the name of the file you want to read

t = Tiff(fname,'r');
info=imfinfo(fname);
nIm=length(info);
l=t.getTag('ImageLength');
w=t.getTag('ImageWidth');
b=t.getTag('BitsPerSample');
sf=t.getTag('SampleFormat');
switch b
    case 8
        if sf==1
            dt='uint8';
        else
            dt='int8';
        end
    case 16
        if sf==1
            dt='uint16';
        else
            dt='int16';
        end
    case 32
        if sf==1
            dt='uint16';
        elseif sf==2
            dt='int16';
        else
            dt='single';
        end
    case 64
        dt='double';
end
I=zeros(l,w,nIm,dt);
for i=1:nIm
    t.setDirectory(i);
    I(:,:,i)=t.read();
end
try
    if ~isempty(info(1).Software)
        pMax=strfind(info(1).Software,'Max');
        if ~isempty(pMax)
            dummy=sscanf(info(1).Software,['%*' num2str(pMax+4) 'c %f %*c']);
            m(1)=dummy(1);
        else
            m(1)=1;
        end
        pMin=strfind(info(1).Software,'Min');
        if ~isempty(pMin)
            dummy=sscanf(info(1).Software,['%*' num2str(pMin+4) 'c %f %*c']);
            m(2)=dummy(1);
        else
            m(2)=0;
        end
        I=double(I);
        I=(I-min(I(:)))/(max(I(:))-min(I(:)));
        I=m(2)+I*(m(1)-m(2));
    end
catch err
end
t.close;

end