%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Nicholas Heredia, code begat 09/10/2019
% Computational Intelligence, HW 4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function main(null)
try
    datRaw=load('~\03_NumRecognition2\usps_modified.mat');
catch
    warning("File failed to load. Opening File Explorer");
    datRaw=uigetfile('*.mat'); end

myDat=datRaw.data;

%%=================Variable Declarations=================%%
E_cv=zeros(1,5);     lambda=[0.001,0.01,0.1,0.25,0.5,0.75];
E_in=zeros(6,10);       Eout=zeros(6,10);
wt_Avg = zeros(60,10);

%%==============Feature Extraction and Calc==============%%
[x,y]=getfeatures(myDat);
Iota=[x(:,1),x(:,2),x(:,1).^2,x(:,1).*x(:,2),x(:,2).^2,x(:,1).^3,...
    x(:,1).^2.*x(:,2),x(:,1).*x(:,2).^2,x(:,2).^3];
N=length(x);

figure(1);
hold on
title('Ones versus Other Digits');
scatter(x((y==1),1),x((y==2),2),'ro');
scatter(x((y~=1),1),x((y~=2),2),'g+');
legend('Ones','Not Ones');xlabel('Intensity'); ylabel('Symmetry');
hold off

%%=================Linear Regularization=================%%

for i=1:6   %Increment through lambda for testing
    ptr=randperm(N)';
    for j=1:10 %Ten fold classification
        selxn=ptr([1:4500]);
        xTrial=zeros(4500,3); xTrial(:,1)=1;
        xTrial(:,2:3)=x(selxn,:);
        ioTrial=Iota(selxn,:);
        %yTrial=y(selxn,:); %set to be modified, DNE
        yTrial=sign(rand(4500,1)-0.5);
        [w] = linreg( ioTrial,yTrial,lambda(i));
        wSize=size(w);
        %[w] = linreg( xTrial(:,2:3),yTrial,lambda(i));
        ioTrial=[ones(4500,1),ioTrial];
        yTrial=sign(ioTrial'.*w)';
        %yTrial=sign(xTrial'.*w)';

        %Set all non-ones to -1
        yTrue=y(ptr,:);
        yTrue(find(yTrue~=1))=-1;
        
        % Calculate Error In
        E_in(i,j)=nnz(yTrial(:,1)-yTrue(1:4500,1)); % Records Errin for each 'bucket'
        
        % Calculate Error out, last decile of  data set
        selxnIn=ptr([4501:end]);
        IoTrialOut=zeros(500,10); IoTrialOut(:,1)=1;
        IoTrialOut(:,2:10)=Iota(selxnIn,:);
        yTrialOut=sign(IoTrialOut'.*w)';
        
        Eout(i,j)=nnz(yTrialOut(:,1)-yTrue(4501:5000,1));
        
        %---------------Reorders Data block for next iteration in trial
        xgroup=ptr(1:500,:);            % Selected first 500 to copy
        ptr(1:4500,:)=ptr(501:5000,:); % Shifts 501:5000 elements down to first
        ptr(4501:5000,:)=xgroup;        % Pastes copied group to end
        
        
        wt_Avg(i*wSize(1)-wSize(1)+1:i*wSize(1),j)=w;
    end
    
    wt_Avg(i*wSize(1)-wSize(1)+1:i*wSize(1),1)=...
        mean(wt_Avg(i*wSize(1)-wSize(1)+1:i*wSize(1),1:10),1);
end

Lambda = {'0.001','0.01','0.1','0.25','0.5','0.75'};
AvgErrIn = [mean(E_in(1,:)),mean(E_in(2,:)),mean(E_in(3,:)),...
    mean(E_in(4,:)),mean(E_in(5,:)),mean(E_in(6,:))]
AvgErrOt = [mean(Eout(1,:)),mean(Eout(2,:)),mean(Eout(3,:)),...
    mean(Eout(4,:)),mean(Eout(5,:)),mean(Eout(6,:))]


bestLamb=lambda(find(min(AvgErrIn)));
indBestL=find(lambda==bestLamb);
%yTrial=sign(rand(4500,1)-0.5);
%[w] = linreg( xTrial(:,2:3),yTrial,bestLamb);
w_Fin=wt_Avg(indBestL:indBestL+wSize(1)-1);
yF = sign(ioTrial'.*w)';

figure(2);
hold on
title('Classification of Digits');
for n =1:size(x,1)
    if (yF(n)==1), scatter(x(n,1),x(n,2),'ro'); %ones
    else,         scatter(x(n,1),x(n,2),'g+'); %five
    end
end
xlabel('Intensity'); ylabel('Symmetry');
hold off

end

