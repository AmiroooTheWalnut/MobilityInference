clc
clear

USADistancesByOrder = readmatrix('USA_DistancesByOrder.csv');
USANumVisitsByOrder = readmatrix('USA_NumVisitsByOrder.csv');

numShops=size(USADistancesByOrder,2);

alpha_agg(1,size(USADistancesByOrder,2)-1)=0;
beta(1,size(USADistancesByOrder,2)-1)=0;
gamma(1,size(USADistancesByOrder,2))=0;
for i=1:size(USADistancesByOrder,2)-1
    alpha_agg(1,i)=USADistancesByOrder(1,i+1)/USADistancesByOrder(1,i);
    beta(1,i)=USANumVisitsByOrder(1,i+1)/USANumVisitsByOrder(1,i);
    gamma(1,i)=USADistancesByOrder(1,i)/USANumVisitsByOrder(1,i);
end

USA_DistancesByOrderCBG = readmatrix('USA_DistancesByOrderCBG.csv');
distancesStd=std(USA_DistancesByOrderCBG);
USA_ShopNumVisitsByOrderCBG = readmatrix('USA_NumVisitsByOrderCBG.csv');
numbersStd=std(USA_ShopNumVisitsByOrderCBG);

alpha(size(USA_DistancesByOrderCBG,1),size(USA_DistancesByOrderCBG,2)-1)=0;

for i=1:size(USA_DistancesByOrderCBG,1)
    for j=1:size(USA_DistancesByOrderCBG,2)-1
        alpha(i,j)=USA_DistancesByOrderCBG(i,j+1)/USA_DistancesByOrderCBG(i,j);
    end
end


k_1=[1,1.1];
getGamma(k_1,alpha,USA_ShopNumVisitsByOrderCBG)

k_1=[1,1.5];
getGamma(k_1,alpha,USA_ShopNumVisitsByOrderCBG)

k_1=[1,2];
getGamma(k_1,alpha,USA_ShopNumVisitsByOrderCBG)

k_1=[1,5];
getGamma(k_1,alpha,USA_ShopNumVisitsByOrderCBG)

numBins=12;

ranges=linspace(1,3,numBins);
gammaValuesStacked(size(USA_ShopNumVisitsByOrderCBG,2)-1,size(ranges,2)-1)=0;
for i=1:size(ranges,2)-1
    %     if i==5
    %         disp(12)
    %     end
    gammaXAlphaStacked(i)=(ranges(i)+ranges(i+1))/2;
    gammaValuesStacked(:,i)=getGammaPrime([ranges(i),ranges(i+1)],alpha,USA_ShopNumVisitsByOrderCBG);
end
figure(7)
clf
maxOrder=6;

legendLabels=cell(1,maxOrder);
gammaXAlphaStackedCell=cell(1,size(ranges,2)-1);

for i=1:size(ranges,2)-1
    gammaXAlphaStackedCell{1,i}=strcat('[',num2str(round(ranges(i),2)),',',num2str(round(ranges(i+1),2)),']');
end

for i=1:maxOrder
    legendLabels{1,i}=num2str(i);
end
b=bar(categorical(gammaXAlphaStackedCell),gammaValuesStacked(1:maxOrder,:),'stacked');
allYValues=[];
for i=1:maxOrder
    xtips1 = b(i).XEndPoints;
    ytips1 = b(i).YEndPoints;
    allYValues(i,:)=ytips1;
    y_data=[];
    y_data(1,numBins-1)=0;
    for j=1:numBins-1
        y_data(1,j)=round(gammaValuesStacked(i,j)/sum(gammaValuesStacked(:,j)),3)*100;
    end
    labels1 = string(y_data);
    for j=1:size(labels1,2)
        if y_data(1,j)<3
            labels1{1,j}='';
        else
            labels1{1,j}=strcat(labels1{1,j},'%');
        end
    end
    
    if i>1
        for j=1:size(labels1,2)
            if allYValues(i,j)-allYValues(i-1,j)<5000000
                labels1{1,j}='';
            end
        end
        ytips1=(allYValues(i,:)+allYValues(i-1,:))./2;
    else
        ytips1=allYValues(1,:)./2;
    end
    
    text(xtips1,ytips1-2000000,labels1,'HorizontalAlignment','center',...
        'VerticalAlignment','bottom')
end
lgd = legend(legendLabels);
title(lgd,'Order of closeness')
xlabel('Alpha range')
ylabel('Number of travels')

figure(8)
clf
gammaValuesStackedFixed(size(gammaValuesStacked,1),size(gammaValuesStacked,2))=0;
for c=1:size(gammaValuesStacked,2)
    sumColumn=sum(gammaValuesStacked(:,c));
    for r=1:size(gammaValuesStacked,1)
        gammaValuesStackedFixed(r,c)=gammaValuesStacked(r,c)/sumColumn;
    end
end
b=bar(categorical(gammaXAlphaStackedCell),gammaValuesStackedFixed*100,'stacked');
ylim([0 100])
ytickformat('percentage')

allYValues=[];
for i=1:size(gammaValuesStackedFixed,1)
    xtips1 = b(i).XEndPoints;
    ytips1 = b(i).YEndPoints;
    allYValues(i,:)=ytips1;
    y_data=[];
    y_data(1,numBins-1)=0;
    for j=1:numBins-1
        y_data(1,j)=round(gammaValuesStacked(i,j)/sum(gammaValuesStacked(:,j)),3)*100;
    end
    labels1 = string(y_data);
    for j=1:size(labels1,2)
        if y_data(1,j)<2
            labels1{1,j}='';
        else
            labels1{1,j}=strcat(labels1{1,j},'%');
        end
    end
    
    if i>1
        for j=1:size(labels1,2)
            if allYValues(i,j)-allYValues(i-1,j)<2.7
                labels1{1,j}='';
            end
        end
        ytips1=(allYValues(i,:)+allYValues(i-1,:))./2;
    else
        ytips1=allYValues(1,:)./2;
    end
    
    text(xtips1,ytips1-1.6,labels1,'HorizontalAlignment','center',...
        'VerticalAlignment','bottom','FontName','times','FontWeight','bold','FontSize',12)
end
h=xlabel('\nu range');
set(h, 'FontSize', 14) 
set(h,'FontWeight','bold')
set(h,'FontName','times')
h=ylabel({'Percentage of travels','sorted by order of closeness from bottom'});
set(h, 'FontSize', 14) 
set(h,'FontWeight','bold')
set(h,'FontName','times')
h=gca; h.XAxis.TickLength = [0 0];
grid minor
ax = gca;
ax.YGrid = 'on';
set(gca,'FontSize',12)
set(gca,'FontWeight','bold')
set(gca,'FontName','times')


% figure(9)
% clf
% b=bar(categorical(gammaXAlphaStackedCell),sum(gammaValuesStacked,1));
% h=gca; h.XAxis.TickLength = [0 0];
% xlabel('Alpha range')
% ylabel('Number of travels')
% grid minor
% ax = gca;
% ax.YGrid = 'on';

%\/\/\/ ALPHA I VS # TRAVELS TO I / # TRAVELS TO I->N_CBG\
%\/\/\/ FILLING MISSING VALUE
REVISED_USA_ShopNumVisitsByOrderCBG(size(USA_ShopNumVisitsByOrderCBG,1),size(USA_ShopNumVisitsByOrderCBG,2))=0;
for i=1:size(USA_ShopNumVisitsByOrderCBG,1)
    counter=1;
    for j=1:size(USA_ShopNumVisitsByOrderCBG,2)
        if USA_ShopNumVisitsByOrderCBG(i,j)~=0
            REVISED_USA_ShopNumVisitsByOrderCBG(i,counter)=USA_ShopNumVisitsByOrderCBG(i,j);
            counter=counter+1;
        end
    end
end
%^^^ FILLING MISSING VALUE
alpha_index_value=2;

for i=1:size(USA_DistancesByOrderCBG,1)
    alpha_i_values(i)=log10(alpha(i,alpha_index_value));
    numinator=sum(REVISED_USA_ShopNumVisitsByOrderCBG(i,1:alpha_index_value));
    denominator=sum(REVISED_USA_ShopNumVisitsByOrderCBG(i,alpha_index_value+1:size(USA_DistancesByOrderCBG,2)));
    y(i)=numinator/denominator;
end
% figure(10)
% clf
% scatter(alpha_i_values,y);
% %[model,performance]=dacefit(alpha_i_values',y',@regpoly0,@corrgauss,1.5);
% %stdshade();
% createFit(alpha_i_values,y);
% xlabel(strcat('Alpha_{',num2str(alpha_index_value),'}(b)'))
% ylabel(strcat('# travel from b to NN_{',num2str(alpha_index_value),'} / # travel from b to NN_i i from ',num2str(alpha_index_value),' to', num2str(numShops)))
% %^^^ ALPHA I VS # TRAVELS TO I / # TRAVELS TO I->N_CBG


revised_gammaValuesStacked=gammaValuesStacked;
for j=1:size(gammaValuesStacked,2)
    val=100;
    for i=1:size(gammaValuesStacked,1)
        if gammaValuesStacked(i,j)==0
            revised_gammaValuesStacked(i,j)=val;
            val=max(val-10,0);
        end
    end
end

projects=dir('../TimedData');

for i=3:size(projects,1)
    if projects(i,1).isdir==1
        mainFolder=cd('..');
        cd('TimedData');
        cd(strcat(projects(i,1).name));
        files=dir(pwd);
        for j=3:size(files,1)
            if contains(files(j,1).name,'CBGShopDists')
                shopDists = readmatrix(files(j,1).name);
                [sSh,ISh]=sort(shopDists,2);

                rawProbabilities=[];
                rawProbabilities(size(revised_gammaValuesStacked,1),size(revised_gammaValuesStacked,2))=0;
                sRP=sum(revised_gammaValuesStacked,1);
                for k=1:size(revised_gammaValuesStacked,2)
                    for m=1:size(revised_gammaValuesStacked,1)
                        rawProbabilities(m,k)=revised_gammaValuesStacked(m,k)/sRP(1,k);
                    end
                end
                parts=split(files(j,1).name,'_');
                subParts=split(parts{size(parts,1),1},'.');
                calcSaveFinalProbability(rawProbabilities,ranges,sSh,ISh,strcat('sourceCBG_shopCBG_probability_',parts{size(parts,1)-1,1},'_',subParts{1,1},'.csv'));
            elseif contains(files(j,1).name,'CBGSchoolDists')
                schoolDists = readmatrix(files(j,1).name);
                [sSch,ISch]=sort(schoolDists,2);

                rawProbabilities=[];
                rawProbabilities(size(revised_gammaValuesStacked,1),size(revised_gammaValuesStacked,2))=0;
                sRP=sum(revised_gammaValuesStacked,1);
                for k=1:size(revised_gammaValuesStacked,2)
                    for m=1:size(revised_gammaValuesStacked,1)
                        rawProbabilities(m,k)=revised_gammaValuesStacked(m,k)/sRP(1,k);
                    end
                end
                parts=split(files(j,1).name,'_');
                subParts=split(parts{size(parts,1),1},'.');
                calcSaveFinalProbability(rawProbabilities,ranges,sSch,ISch,strcat('sourceCBG_schoolCBG_probability',parts{size(parts,1)-1,1},'_',subParts{1,1},'.csv'));
            elseif contains(files(j,1).name,'CBGReligionDists')
                religionDists = readmatrix(files(j,1).name);
                [sRel,IRel]=sort(religionDists,2);

                rawProbabilities=[];
                rawProbabilities(size(revised_gammaValuesStacked,1),size(revised_gammaValuesStacked,2))=0;
                sRP=sum(revised_gammaValuesStacked,1);
                for k=1:size(revised_gammaValuesStacked,2)
                    for m=1:size(revised_gammaValuesStacked,1)
                        rawProbabilities(m,k)=revised_gammaValuesStacked(m,k)/sRP(1,k);
                    end
                end
                parts=split(files(j,1).name,'_');
                subParts=split(parts{size(parts,1),1},'.');
                calcSaveFinalProbability(rawProbabilities,ranges,sRel,IRel,strcat('sourceCBG_religionCBG_probability',parts{size(parts,1)-1,1},'_',subParts{1,1},'.csv'));
            end
        end
        cd(mainFolder);
    end
end

% shopDists = readmatrix('USA_NM_Santa Fe County_Santa Fe_CBGShopDists.csv');
% schoolDists = readmatrix('USA_NM_Santa Fe County_Santa Fe_CBGSchoolDists.csv');
% religionDists = readmatrix('USA_NM_Santa Fe County_Santa Fe_CBGReligionDists.csv');
% 
% [sSh,ISh]=sort(shopDists,2);
% [sSch,ISch]=sort(schoolDists,2);
% [sRel,IRel]=sort(religionDists,2);
% 
% rawProbabilities(size(revised_gammaValuesStacked,1),size(revised_gammaValuesStacked,2))=0;
% sRP=sum(revised_gammaValuesStacked,1);
% for j=1:size(revised_gammaValuesStacked,2)
%     for i=1:size(revised_gammaValuesStacked,1)
%         rawProbabilities(i,j)=revised_gammaValuesStacked(i,j)/sRP(1,j);
%     end
% end
% calcSaveFinalProbability(rawProbabilities,ranges,sSh,ISh,'sourceCBG_shopCBG_probability.csv');
% calcSaveFinalProbability(rawProbabilities,ranges,sSch,ISch,'sourceCBG_schoolCBG_probability.csv');
% calcSaveFinalProbability(rawProbabilities,ranges,sRel,IRel,'sourceCBG_religionCBG_probability.csv');


function calcSaveFinalProbability(rawProbabilities,ranges,sortedVals,sortedIndex,filename)
finalProbabilities(size(sortedVals,1),size(sortedVals,2))=0;
for i=1:size(sortedVals,1)
    remaining_prob=1;
    for j=1:size(sortedVals,2)-1
        seletedAlphaIndex=0;
        if sortedVals(i,j)==0 && sortedVals(i,j+1)==0
            seletedAlphaIndex=1;
            finalProbabilities(i,sortedIndex(i,j))=rawProbabilities(j,seletedAlphaIndex)*remaining_prob;
            remaining_prob=remaining_prob-rawProbabilities(j,seletedAlphaIndex);
        elseif sortedVals(i,j)==0
            seletedAlphaIndex=1;
            finalProbabilities(i,sortedIndex(i,j))=rawProbabilities(j,seletedAlphaIndex)*remaining_prob;
            remaining_prob=remaining_prob-rawProbabilities(j,seletedAlphaIndex);
        else
            for m=2:size(ranges,2)
                if sortedVals(i,j+1)/sortedVals(i,j)<ranges(1,m)
                    seletedAlphaIndex=m-1;
                    break;
                end
            end
            if seletedAlphaIndex==0
                seletedAlphaIndex=size(ranges,2)-1;
            end
            if j<=size(rawProbabilities,1)
                finalProbabilities(i,sortedIndex(i,j))=rawProbabilities(j,seletedAlphaIndex)*remaining_prob;
                remaining_prob=remaining_prob-rawProbabilities(j,seletedAlphaIndex);
            else
                finalProbabilities(i,sortedIndex(i,j))=finalProbabilities(i,sortedIndex(i,size(rawProbabilities,1)));
            end
        end
    end
    finalProbabilities(i,sortedIndex(i,size(sortedVals,2)))=finalProbabilities(i,sortedIndex(i,size(rawProbabilities,1)));
end

for i=1:size(finalProbabilities,1)
    minV=min(finalProbabilities(i,:));
    maxV=max(finalProbabilities(i,:));
    for j=1:size(finalProbabilities,2)
        finalProbabilities(i,j)=(finalProbabilities(i,j)-minV+(maxV-minV)*0.05)/(maxV+(maxV-minV)*0.05);
    end
    S=sum(finalProbabilities(i,:));
    for j=1:size(finalProbabilities,2)
        finalProbabilities(i,j)=finalProbabilities(i,j)/S;
    end
end

writematrix(finalProbabilities,filename)
end

function gamma=getGamma(k,alpha,USA_ShopNumVisitsByOrderCBG)
popleGoToN1_k=0;
popleGoToN2To70_k=0;
for i=1:size(USA_ShopNumVisitsByOrderCBG,1)
    for j=1:size(USA_ShopNumVisitsByOrderCBG,2)-1
        if alpha(i,j)>k(1,1) && alpha(i,j)<k(1,2)
            if j==1
                popleGoToN1_k=popleGoToN1_k+USA_ShopNumVisitsByOrderCBG(i,j);
            else
                popleGoToN2To70_k=popleGoToN2To70_k+USA_ShopNumVisitsByOrderCBG(i,j);
            end
        end
    end
end
popleGoToN1=0;
popleGoToN2To70=0;
for i=1:size(USA_ShopNumVisitsByOrderCBG,1)
    for j=1:size(USA_ShopNumVisitsByOrderCBG,2)-1
        if j==1
            popleGoToN1=popleGoToN1+USA_ShopNumVisitsByOrderCBG(i,j);
        else
            popleGoToN2To70=popleGoToN2To70+USA_ShopNumVisitsByOrderCBG(i,j);
        end
    end
end

gamma=((popleGoToN1_k)/(popleGoToN2To70_k))/((popleGoToN1)/(popleGoToN2To70));
end


function gamma=getGammaPrime(k,alpha,USA_ShopNumVisitsByOrderCBG)
popleGoToNi_k(1,size(USA_ShopNumVisitsByOrderCBG,2)-1)=0;
for i=1:size(USA_ShopNumVisitsByOrderCBG,1)
    for j=1:size(USA_ShopNumVisitsByOrderCBG,2)-1
        if alpha(i,j)>k(1,1) && alpha(i,j)<k(1,2)
            popleGoToNi_k(1,j)=popleGoToNi_k(1,j)+USA_ShopNumVisitsByOrderCBG(i,j);
        end
    end
end

gamma=popleGoToNi_k';
end