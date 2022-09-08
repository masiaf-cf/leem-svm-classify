function [ Z,iter,ok ] = solveNormalEqComb( AtA,AtB,PassSet )
% Solve normal equations using combinatorial grouping.
% Although this function was originally adopted from the code of
% "M. H. Van Benthem and M. R. Keenan, J. Chemometrics 2004; 18: 441-450",
% important modifications were made to fix bugs.
%
% Modified by Jingu Kim (jingu@cc.gatech.edu)
%             School of Computational Science and Engineering,
%             Georgia Institute of Technology
%
% Last updated Aug-12-2009
ok=1;
iter = 0;

if (nargin ==2) || isempty(PassSet) || all(PassSet(:))
    countsing=0;
    Z = AtA\AtB;
%     if isnan(rcond(AtA))
%         ok=0;
%         return
%     end
    iter = iter + 1;
    %Modified by FM on 19/08/2014
    [~, msgidlast] = lastwarn;
    while strcmp(msgidlast,'MATLAB:nearlySingularMatrix')||strcmp(msgidlast,'MATLAB:SingularMatrix')||strcmp(msgidlast,'MATLAB:singularMatrix')||strcmp(msgidlast,'MATLAB:illConditionedMatrix')
        lastwarn('');
        AtAb=AtA+rand(size(AtA))*0.01*norm(AtA,'fro');
        Z=AtAb\AtB;
        [~, msgidlast] = lastwarn;
        countsing=countsing+1;
        if countsing>100
            ok=0;
            return;
        end
    end
else
    Z = zeros(size(AtB));
    [n,k1] = size(PassSet);
    
    %% Fixed on Aug-12-2009
    if k1==1
        countsing=0;
        Z(PassSet)=AtA(PassSet,PassSet)\AtB(PassSet);
%         if isnan(rcond(AtA(PassSet,PassSet)))
%             ok=0;
%             return
%         end
        [~, msgidlast] = lastwarn;
        while strcmp(msgidlast,'MATLAB:nearlySingularMatrix')||strcmp(msgidlast,'MATLAB:SingularMatrix')||strcmp(msgidlast,'MATLAB:singularMatrix')||strcmp(msgidlast,'MATLAB:illConditionedMatrix')
            lastwarn('');
            AtAb=AtA(PassSet,PassSet)+rand(size(AtA(PassSet,PassSet)))*0.01*norm(AtA(PassSet,PassSet),'fro');
            Z(PassSet)=AtAb\AtB(PassSet);
            [~, msgidlast] = lastwarn;
            countsing=countsing+1;
            if countsing>100
                ok=0;
                return;
            end
        end
    else
        %% Fixed on Aug-12-2009
        % The following bug was identified by investigating a bug report by Hanseung Lee.
        [sortedPassSet,sortIx] = sortrows(PassSet');
        breaks = any(diff(sortedPassSet)');
        breakIx = [0 find(breaks) k1];
        % codedPassSet = 2.^(n-1:-1:0)*PassSet;
        % [sortedPassSet,sortIx] = sort(codedPassSet);
        % breaks = diff(sortedPassSet);
        % breakIx = [0 find(breaks) k1];
        
        for k=1:length(breakIx)-1
            countsing=0;
            cols = sortIx(breakIx(k)+1:breakIx(k+1));
            vars = PassSet(:,sortIx(breakIx(k)+1));
            Z(vars,cols) = AtA(vars,vars)\AtB(vars,cols);
%             if isnan(rcond(AtA(vars,vars)))
%                 ok=0;
%                 return
%             end
            iter = iter + 1;
            %Modified by FM on 19/08/2014
            [~, msgidlast] = lastwarn;
            while strcmp(msgidlast,'MATLAB:nearlySingularMatrix')||strcmp(msgidlast,'MATLAB:SingularMatrix')||strcmp(msgidlast,'MATLAB:singularMatrix')||strcmp(msgidlast,'MATLAB:illConditionedMatrix')
                lastwarn('');
                AtAb=AtA(vars,vars)+rand(size(AtA(vars,vars)))*0.01*norm(AtA(vars,vars),'fro');
                Z(vars,cols) = AtAb\AtB(vars,cols);
                [~, msgidlast] = lastwarn;
                countsing=countsing+1;
                if countsing>100
                    ok=0;
                    return;
                end
            end
            
        end
    end
end
end
