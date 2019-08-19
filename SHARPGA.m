function [Psif,Xf,Tf,Fitness,Penalty]=SHARPGA(Net_Time,ActUtil,ActDur,Coeff,Ta,Tb,ArrGoal,ArrPen,HomeLoc,ActLoc,k,P,G,mutprob)
 
%SHARPGA - GA for solving Generalized Selective Household Activity Routing Problem, which extends Recker’s (1995) HAPP model 
% to include activity destination choice (with and without quota for activity types -- 
%for example, work can be modeled as a required k=1 activity type which would select the location that offers the best utility 
%(set of choices can be limited if short term evaluation is desired without choice of employment location), and shopping can be done 
%with k=0 where there is no limit to the number of destinations visited, so long as marginal utility gained is greater than marginal cost of tour)
%This is the benchmark algorithm that I used to compare the reoptimization algorithm against in the paper:
%“Activity-based travel scenario analysis with routing problem reoptimization”

%This particular implementation is set up with 2 work shifts (morning (PA1 – primary activity shift 1) and afternoon (PA2 – primary activity shift 2)), but if that is not desired it can be modified to do so
%INPUTS:
   %Net_Time = (Nsum+2)x(Nsum+2) travel time matrix, for Nsum activities
   %plus origin and final destination; sequenced as [O Activities D]
   %ActUtil = is a cell of {Mx1} with N_m conditional utilities of each activity given activity
   %type m E M activity types chosen
   %I will automatically force m = 2 to be PA2 and m = 1 to be PA1, and set
   %to same location
   %ActDur = Mx1 durations of each activity type desired
   %Coeff = (M+1)x1 coefficients for each activity type plus length of tour
   %minus activity durations
   %[Ta,Tb] = 2 (2*Nsum+2)x1 vectors representing arrival time windows for all activity types plus final destination - if null, then
   %ArrGoal = Mx1 goal arrival time per type m
   %ArrPen = Mx2, early penalty and late penalty per type m
   %k is Mx1 number of activities per type, (fixed types are assigned k=1)
%OUTPUTS: (as defined in the paper)
    %Psif = 2*Nsum choices of activities per activity type
    %Xf = routes chosen by traveler
    %Tf = arrival times at each node by traveler
    %Fitness = the fitness value of the objective
    %Penalty = penalty of a solution due to infeasibility
  
   
M=size(ActDur,1); %number of activity types
N=zeros(M,1); %number of activities
for i=1:M
    N(i,1)=size(ActUtil{i,1},1);
end
Nsum=sum(N);
options=optimset('Display','off');
 
%for translating to activity type from size N
Nm=zeros(2*Nsum,2);
m=1;
for i=1:Nsum
    Nm(i,1)=i;
    if i<=sum(N(1:m,1))
        Nm(i,2)=m;
    else
        Nm(i,2)=m+1;
        m=m+1;
    end
end
Nm(Nsum+1:2*Nsum,1)=Nm(1:Nsum,1);
Nm(Nsum+1:2*Nsum,2)=Nm(1:Nsum,2)+M;
 
%Constructing the objective function
f=zeros(1,2*Nsum+(2*Nsum+2)*(2*Nsum+2)+2*Nsum+2+2*Nsum);
i=0;
for m=1:M
    for u=1:N(m,1)
        i=i+1;
        f(1,i)=ActUtil{m,1}(u,1);
    end
end
 
 
f(1,2*Nsum+1:2*Nsum+(2*Nsum+2)*(2*Nsum+2))=-Coeff(1,1);
f(1,2*Nsum+(2*Nsum+2)^2+Nsum+1:2*Nsum+(2*Nsum+2)^2+2*Nsum)=-Coeff(2,1);
f(1,2*Nsum+(2*Nsum+2)^2+1:2*Nsum+(2*Nsum+2)^2+Nsum)=Coeff(2,1);
f(1,2*Nsum+(2*Nsum+2)^2+2*Nsum+1)=-Coeff(3,1);
f(1,2*Nsum+(2*Nsum+2)^2+2*Nsum+2)=Coeff(3,1);
 
for j=1:Nsum
    f(1,2*Nsum+(2*Nsum+2)*(2*Nsum+2)+2*Nsum+2+j)=-ArrPen(Nm(j,2),1);
    f(1,2*Nsum+(2*Nsum+2)*(2*Nsum+2)+2*Nsum+2+Nsum+j)=-ArrPen(Nm(j,2),2);
end
 
%GA Heuristic
%initiate
 
%LHS
xmin=zeros(1,Nsum);
xmax=ones(1,Nsum);
nvarLHS=length(xmin);
ran=rand(P,nvarLHS);
s=zeros(P,nvarLHS);
s1=zeros(P,nvarLHS);
for j=1: nvarLHS
   idx=randperm(P);
   temp =(idx'-ran(:,j))/P;
   s(:,j) = xmin(j) + temp.* (xmax(j)-xmin(j));
   for p=1:P
      s1(p,j)=rand;
   end
end
 
%create initial population - consider fixed sequence (k=1), k>0, k=0
PenVal=10000;
Psif=zeros(2*Nsum,P);
Tf=zeros(2*Nsum+2,P);
Xf=zeros((2*Nsum+2)^2,P); 
Et=zeros(Nsum,P);
Lt=zeros(Nsum,P);
 
Penalty=zeros(P,G);
Fitness=zeros(P,G)-inf;
%choose nodes to visit first
for m=1:M
    if k(m,1)==0
        for i=sum(N(1:m-1,1))+1:sum(N(1:m,1))
            for p=1:P
                if s(p,i)>0.98
                    Psif(i,p)=1;
                    Psif(i+Nsum,p)=1;
                end
            end
        end
    else
        for p=1:P
            if k(m,1)>1
                unsortedlist=s(p,sum(N(1:m-1,1))+1:sum(N(1:m,1)));
                [~,sortedlist]=sort(unsortedlist,'descend');
                for kk=1:k(m,1)
                    Psif(sum(N(1:m-1,1))+sortedlist(1,kk),p)=1;
                    Psif(sum(N(1:m-1,1))+sortedlist(1,kk)+Nsum,p)=1;
                end
            else
                if m>2
                    maxi=1;
                    for kk=2:N(m,1)
                        if s(p,sum(N(1:m-1,1))+kk)>s(p,sum(N(1:m-1,1))+maxi)
                            maxi=kk;
                        end
                    end
                    Psif(sum(N(1:m-1,1))+maxi,p)=1;
                    Psif(sum(N(1:m-1,1))+maxi+Nsum,p)=1;
                elseif m==1 %this forces m=1 (PA1) and m=2 (PA2) to have the same location
                    maxi=1;
                    for kk=2:N(m,1)
                        if s(p,sum(N(1:m-1,1))+kk)>s(p,sum(N(1:m-1,1))+maxi)
                            maxi=kk;
                        end
                    end
                    Psif(sum(N(1:m-1,1))+maxi,p)=1;
                    Psif(sum(N(1:m,1))+maxi,p)=1;
                    Psif(sum(N(1:m-1,1))+maxi+Nsum,p)=1;
                    Psif(sum(N(1:m,1))+maxi+Nsum,p)=1;
                end
            end
        end
    end
end
 
%choose route through nodes
route=zeros(Nsum,P);
sequence=zeros(2*Nsum,P); %doubled the sequence length for dropoffs
for p=1:P
    route(:,p)=Nm(1:Nsum,1);
    [~,IX]=sort(s1(p,:));
    route(:,p)=IX';
    [~,IX]=sort(route(:,p));
    
    %update sequence with dropoffs
    tempseq=IX;
    firstreturn=Nsum;
    for j=1:Nsum
        temp=firstreturn-Nsum+j+ceil(rand*(size(tempseq,1)-Nsum+j-(firstreturn-Nsum+j))); 
        tempseq=[tempseq(1:(Nsum-j+temp),1); tempseq(Nsum-j+1,1)+Nsum; tempseq(Nsum-j+temp+1:size(tempseq,1),1)];
        firstreturn=temp+Nsum-j;
    end
    sequence(:,p)=tempseq;
    
    %check for redundant sequences
    sameseq=0;
    q=1;
    while and(q<=p-1,sameseq==0)
        if sum(sequence(:,p).*Psif(:,p)~=sequence(:,q).*Psif(:,q))==0
            sameseq=1;
        else
            q=q+1;
        end
    end
    if sameseq==0
        origin=1;
        NumSeq=sum(Psif(:,p));
        etime=zeros(2*Nsum,1); %early penalty
        ltime=zeros(2*Nsum,1);
        Atime=zeros(2*Nsum,NumSeq*2+2);
        btime=zeros(2*Nsum,1);
        TSeq=zeros(2*Nsum,1);
        ineq=0;
 
        for j=1:2*Nsum
            if Psif(sequence(j,p),p)==1
                ineq=ineq+1;
                Xf((origin-1)*(2*Nsum+2)+sequence(j,p)+1,p)=1;
                if sequence(j,p)<=Nsum
                    etime(ineq,1)=ArrPen(Nm(sequence(j,p),2),1);
                    ltime(ineq,1)=ArrPen(Nm(sequence(j,p),2),2);
                end
                TSeq(ineq,1)=sequence(j,p);
                if origin==1
                    Atime(ineq,NumSeq*2+1)=1;
                    Atime(ineq,ineq)=-1;
                    Atime(ineq,NumSeq+ineq)=1;
                    btime(ineq,1)=ArrGoal(Nm(sequence(j,p),2),1)-Net_Time(HomeLoc,ActLoc(sequence(j,p),1));
                else
                    Atime(ineq,ineq-1)=1;
                    Atime(ineq,NumSeq+ineq-1)=-1;
                    Atime(ineq,ineq)=-1;
                    Atime(ineq,NumSeq+ineq)=1;
                    if and(sequence(j,p)<=Nsum,origin<=Nsum+1)
                        btime(ineq,1)=ArrGoal(Nm(sequence(j,p),2),1)-ArrGoal(Nm(origin-1,2),1)-ActDur(Nm(origin-1,2),1)-Net_Time(ActLoc(origin-1,1),ActLoc(sequence(j,p),1));
                    elseif and(sequence(j,p)>Nsum,origin<=Nsum+1)
                        btime(ineq,1)=ArrGoal(Nm(sequence(j,p),2),1)-ArrGoal(Nm(origin-1,2),1)-ActDur(Nm(origin-1,2),1)-Net_Time(ActLoc(origin-1,1),HomeLoc);
                    elseif and(sequence(j,p)<=Nsum,origin>Nsum+1)
                        btime(ineq,1)=ArrGoal(Nm(sequence(j,p),2),1)-ArrGoal(Nm(origin-1,2),1)-Net_Time(HomeLoc,ActLoc(sequence(j,p),1));
                    else
                        btime(ineq,1)=ArrGoal(Nm(sequence(j,p),2),1)-ArrGoal(Nm(origin-1,2),1);
                    end
                end
                origin=sequence(j,p)+1;
            end
        end
        
        if origin~=1
            Xf((origin-1)*(2*Nsum+2)+2*Nsum+2,p)=1;
            Atime(ineq+1,ineq)=1;
            Atime(ineq+1,NumSeq+ineq)=-1;
            Atime(ineq+1,2*NumSeq+2)=-1;
            btime(ineq+1,1)=-ArrGoal(Nm(origin-1,2),1);
        end
 
        etime=etime(1:ineq,:);
        ltime=ltime(1:ineq,:);
        Atime=Atime(1:ineq+1,:);
        btime=btime(1:ineq+1,:);
        TSeq=TSeq(1:ineq,:);
        
        %solve for the Tfs
        Tsol=linprog([ltime; etime; -Coeff(3,1); Coeff(3,1)],Atime,btime,[],[],zeros(2*NumSeq+2,1),[],[],options);
        for j=1:NumSeq
            Tf(TSeq(j,1),p)=ArrGoal(Nm(TSeq(j,1),2),1)+Tsol(j,1)-Tsol(NumSeq+j,1);
            if TSeq(j,1)<=Nsum
                Et(TSeq(j,1),p)=Tsol(NumSeq+j,1);
                Lt(TSeq(j,1),p)=Tsol(j,1);
            end
        end
        Tf(2*Nsum+1,p)=Tsol(2*NumSeq+1,1);
        Tf(2*Nsum+2,p)=Tsol(2*NumSeq+2,1);
    
        %determine fitness
        if size(Tb,1)>0
            Penalty(p,1)=PenVal*(sum(max(Tf(:,p)-Tb,0))+sum(max(Ta-Tf(:,p),0)));
        end
        Fitness(p,1)=f*[Psif(:,p);Xf(:,p);Tf(:,p);Et(:,p);Lt(:,p)]-Penalty(p,1);
    else
        Xf(:,p)=Xf(:,q);
        Tf(:,p)=Tf(:,q);
        Et(:,p)=Et(:,q);
        Lt(:,p)=Lt(:,q);
        Penalty(p,1)=Penalty(q,1);
        Fitness(p,1)=Fitness(q,1);
    end
end
 
for g=2:G
    %crossbreed
    [~,FIX]=sort(Fitness(:,g-1),'descend');
    newseq=zeros(2*Nsum,P);
    newpsi=zeros(2*Nsum,P);
    newx=zeros((2*Nsum+2)^2,P);
    newt=zeros(2*Nsum+2,P);
    newet=zeros(Nsum,P);
    newlt=zeros(Nsum,P);
    newpen=zeros(P,1);
    newfitness=zeros(P,1);
    for p=1:P
        store=randperm(P/2);  
        parent1=sequence(:,FIX(store(1),1));
        parent2=sequence(:,FIX(store(2),1));
        newpsi(:,p)=Psif(:,FIX(store(2)));
        store2=randperm(2*Nsum);
        start=min(store2(1:2));
        finish=max(store2(1:2));
        parent2string=parent2;
        for i=start:finish
            j=1;
            while j<=size(parent2string,1)
                if parent2string(j,1)==parent1(i,1)
                    parent2string=[parent2string(1:j-1,1); parent2string(j+1:size(parent2string,1),1)];
                    newpsi(i,p)=Psif(i,FIX(store(1)));
                end
                j=j+1;
            end
        end
        newseq(:,p)=[parent2string(1:start-1,1); parent1(start:finish,1); parent2string(start:size(parent2string,1),1)];
        
        %mutate - for fixed k, we just mutate by adding/deleting back to k, and
        %then modifying the remainder with probability mutprob
        for m=1:M
            if k(m,1)>0
                %set back to correct number
                sumk=sum(newpsi(sum(N(1:m-1,1))+1:sum(N(1:m,1)),p));
                if sumk>k(m,1) %need to remove
                    store2=randperm(sumk);
                    store2=store2(1:sumk-k(m,1));
                    count=0;
                    for i=sum(N(1:m-1,1))+1:sum(N(1:m,1))
                        if newpsi(i,p)==1
                            count=count+1;
                            for j=1:sumk-k(m,1)
                                if store2(j)==count
                                    newpsi(i,p)=0;
                                end
                            end
                        end
                    end
                elseif sumk<k(m,1) %need to add
                    store2=randperm(N(m,1)-sumk);
                    store2=store2(1:k(m,1)-sumk);
                    count=0;
                    for i=sum(N(1:m-1,1))+1:sum(N(1:m,1))
                        if newpsi(i,p)==0
                            count=count+1;
                            for j=1:k(m,1)-sumk
                                if store2(j)==count
                                    newpsi(i,p)=1;
                                end
                            end
                        end
                    end
                end
            end
        end
        %now we randomly replace
        for j=1:2*Nsum
            if and(newpsi(newseq(j,p),p)==1,newseq(j,p)<=Nsum)
                mutateseed=rand;
                if mutateseed<mutprob/(sum(k,1)+1)  
                    count=ceil(rand*2*Nsum);
                    stop=0;
                    while stop==0
                        if and(and(newpsi(newseq(count,p),p)==0,Nm(newseq(count,p),2)==Nm(newseq(j,p),2)),newseq(count,p)<=Nsum)
                            newpsi(newseq(count,p),p)=1;
                            newpsi(newseq(j,p),p)=0;
                            tempseq=newseq(count,p);
                            newseq(count,p)=newseq(j,p);
                            newseq(j,p)=tempseq;
                            stop=1;
                        else
                            count=count+1;
                            if count>2*Nsum
                                count=mod(count,2*Nsum);
                            end
                        end
                    end
                end
            end
        end
        
        %ensure that 2nd shift in work (m=2) is the same location as 1st
        %shift (m=1)
        m=2;
        newpsi(sum(N(1:m-1,1))+1:sum(N(1:m,1)),p)=0;
        for j=1:N(m-1,1)
            if newpsi(sum(N(1:m-2,1))+j,p)==1
                newpsi(sum(N(1:m-1,1))+j,p)=1;
            end
        end
        
        %correct the dropoffs' psi values
        for j=1:Nsum
            if newpsi(j,p)==1
                newpsi(j+Nsum,p)=1;
            else
                newpsi(j+Nsum,p)=0;
            end
        end
        
        %randomly re-assign dropoffs that occur before pickups
        subseq=zeros(2*Nsum,1);
        tempseq=newseq(:,p);
        count=0;
        for j=1:2*Nsum
            if and(tempseq(2*Nsum-j+1,1)>Nsum,2*Nsum-j+1<find(tempseq(:,1)==tempseq(2*Nsum-j+1,1)-Nsum))
                count=count+1;
                subseq(count,1)=tempseq(2*Nsum-j+1,1)-Nsum;
                tempseq=[tempseq(1:2*Nsum-j,1); tempseq(2*Nsum-j+2:size(tempseq,1),1)];
            end
        end
        subseq=subseq(1:count,:);
        tempsum=size(tempseq,1);
        firstreturn=tempsum;
        for j=1:tempsum
            if sum(tempseq(tempsum-j+1,1)==subseq)>0
                temp=firstreturn-tempsum+j+ceil(rand*(size(tempseq,1)-tempsum+j-(firstreturn-tempsum+j))); %fixed 9/19/12
                tempseq=[tempseq(1:(tempsum-j+temp),1); tempseq(tempsum-j+1,1)+Nsum; tempseq(tempsum-j+temp+1:size(tempseq,1),1)];
                firstreturn=temp+tempsum-j;
            end
        end
        newseq(:,p)=tempseq;   
 
        %check for redundant sequences
        sameseq=0;
        sameseq1=0;
        q=1;
        while and(q<=P,sameseq==0)
            if sum(newseq(:,p).*newpsi(:,p)~=sequence(:,q).*Psif(:,q))==0
                sameseq=1;
            else
                q=q+1;
            end
        end
        if sameseq==0
            q1=1;
            while and(q1<=p-1,sameseq1==0)
                if sum(newseq(:,p).*newpsi(:,p)~=newseq(:,q1).*newpsi(:,q1))==0
                    sameseq1=1;
                else
                    q1=q1+1;
                end
            end
        end
        if and(sameseq==0,sameseq1==0)
            %evaluate
            origin=1;
            NumSeq=sum(newpsi(:,p));
            etime=zeros(2*Nsum,1);
            ltime=zeros(2*Nsum,1);
            Atime=zeros(2*Nsum,NumSeq*2+2);
            btime=zeros(2*Nsum,1);
            TSeq=zeros(2*Nsum,1);
            ineq=0;
 
 
            for j=1:2*Nsum
                if newpsi(newseq(j,p),p)==1
                    ineq=ineq+1;
                    newx((origin-1)*(2*Nsum+2)+newseq(j,p)+1,p)=1;
                    if newseq(j,p)<=Nsum
                        etime(ineq,1)=ArrPen(Nm(newseq(j,p),2),1);
                        ltime(ineq,1)=ArrPen(Nm(newseq(j,p),2),2);
                    end
                    TSeq(ineq,1)=newseq(j,p);
                    if origin==1
                        Atime(ineq,NumSeq*2+1)=1;
                        Atime(ineq,ineq)=-1;
                        Atime(ineq,NumSeq+ineq)=1;
                        btime(ineq,1)=ArrGoal(Nm(newseq(j,p),2),1)-Net_Time(HomeLoc,ActLoc(newseq(j,p),1));
                    else
                        Atime(ineq,ineq-1)=1;
                        Atime(ineq,NumSeq+ineq-1)=-1;
                        Atime(ineq,ineq)=-1;
                        Atime(ineq,NumSeq+ineq)=1;
                        if and(newseq(j,p)<=Nsum,origin<=Nsum+1)
                            btime(ineq,1)=ArrGoal(Nm(newseq(j,p),2),1)-ArrGoal(Nm(origin-1,2),1)-ActDur(Nm(origin-1,2),1)-Net_Time(ActLoc(origin-1,1),ActLoc(newseq(j,p),1));
                        elseif and(newseq(j,p)>Nsum,origin<=Nsum+1)
                            btime(ineq,1)=ArrGoal(Nm(newseq(j,p),2),1)-ArrGoal(Nm(origin-1,2),1)-ActDur(Nm(origin-1,2),1)-Net_Time(ActLoc(origin-1,1),HomeLoc);
                        elseif and(newseq(j,p)<=Nsum,origin>Nsum+1)
                            btime(ineq,1)=ArrGoal(Nm(newseq(j,p),2),1)-ArrGoal(Nm(origin-1,2),1)-Net_Time(HomeLoc,ActLoc(newseq(j,p),1));
                        else
                            btime(ineq,1)=ArrGoal(Nm(newseq(j,p),2),1)-ArrGoal(Nm(origin-1,2),1);
                        end
                    end
                    origin=newseq(j,p)+1;
                end
            end
            if origin~=1
                newx((origin-1)*(2*Nsum+2)+2*Nsum+2,p)=1;
                Atime(ineq+1,ineq)=1;
                Atime(ineq+1,NumSeq+ineq)=-1;
                Atime(ineq+1,2*NumSeq+2)=-1;
                btime(ineq+1,1)=-ArrGoal(Nm(origin-1,2),1);
            end
 
            etime=etime(1:ineq,:);
            ltime=ltime(1:ineq,:);
            Atime=Atime(1:ineq+1,:);
            btime=btime(1:ineq+1,:);
            TSeq=TSeq(1:ineq,:);
 
            %solve for the Tfs
            Tsol=linprog([ltime; etime; -Coeff(3,1); Coeff(3,1)],Atime,btime,[],[],zeros(2*NumSeq+2,1),[],[],options);
            for j=1:NumSeq
                newt(TSeq(j,1),p)=ArrGoal(Nm(TSeq(j,1),2),1)+Tsol(j,1)-Tsol(NumSeq+j,1);
                if TSeq(j,1)<=Nsum
                    newet(TSeq(j,1),p)=Tsol(NumSeq+j,1);
                    newlt(TSeq(j,1),p)=Tsol(j,1);
                end
            end
            newt(2*Nsum+1,p)=Tsol(2*NumSeq+1,1);
            newt(2*Nsum+2,p)=Tsol(2*NumSeq+2,1);
 
            %determine fitness
            if size(Tb,1)>0
                newpen(p,1)=PenVal*(sum(max(newt(:,p)-Tb,0))+sum(max(Ta-Tf(:,p),0)));
            end
            newfitness(p,1)=f*[newpsi(:,p);newx(:,p);newt(:,p);newet(:,p);newlt(:,p)]-newpen(p,1);   
        elseif and(sameseq==1,sameseq1==0)
            newx(:,p)=Xf(:,q);
            newt(:,p)=Tf(:,q);
            newet(:,p)=Et(:,q);
            newlt(:,p)=Lt(:,q);
            newpen(p,1)=Penalty(q,1);
            newfitness(p,1)=Fitness(q,1);
        else
            newx(:,p)=newx(:,q1);
            newt(:,p)=newt(:,q1);
            newet(:,p)=newet(:,q1);
            newlt(:,p)=newlt(:,q1);
            newpen(p,1)=newpen(q1,1);
            newfitness(p,1)=newfitness(q1,1);
        end
    end
    
    %merge populations and keep best
    [~,IX]=sort([Fitness(:,g-1); newfitness],'descend');
    p=0;
    extra=0;
    updPsif=zeros(size(Psif,1),P);
    updXf=zeros(size(Xf,1),P);
    updTf=zeros(size(Tf,1),P);
    updseq=zeros(2*Nsum,P);
    while and(p<P+extra,p<2*P)
        p=p+1;
        redundant=0;
        if IX(p,1)<=P
            tempfit=Psif(:,IX(p,1));
        else
            tempfit=newpsi(:,IX(p,1)-P);
        end
        for q=1:p-1-extra
            if updPsif(:,q)==tempfit
                redundant=1;
            end
        end
        if redundant==0
            if IX(p,1)<=P
                Fitness(p-extra,g)=Fitness(IX(p,1),g-1);
                Penalty(p-extra,g)=Penalty(IX(p,1),g-1);
                updPsif(:,p-extra)=Psif(:,IX(p,1));
                updXf(:,p-extra)=Xf(:,IX(p,1));
                updTf(:,p-extra)=Tf(:,IX(p,1));
                updseq(:,p-extra)=sequence(:,IX(p,1));
            else
                Fitness(p-extra,g)=newfitness(IX(p,1)-P,1);
                Penalty(p-extra,g)=newpen(IX(p,1)-P,1);
                updPsif(:,p-extra)=newpsi(:,IX(p,1)-P);
                updXf(:,p-extra)=newx(:,IX(p,1)-P);
                updTf(:,p-extra)=newt(:,IX(p,1)-P);
                updseq(:,p-extra)=newseq(:,IX(p,1)-P);
            end
        else
            extra=extra+1;
        end
    end
    
    %fill in the blanks
    for q=p-extra+1:P
        store=ceil(rand*(p-extra));
        Penalty(q,g)=Penalty(store,g);
        updPsif(:,q)=updPsif(:,store);
        updXf(:,q)=updXf(:,store);
        updTf(:,q)=updTf(:,store);
        updseq(:,q)=updseq(:,store);
        %intentionally leave fitness unupdated so that it's -inf
    end
    
    Psif=updPsif;
    Xf=updXf;
    Tf=updTf;
    sequence=updseq;
end

