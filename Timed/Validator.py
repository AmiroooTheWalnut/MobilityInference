import numpy as np
import logging

def validate(train,tests, elbo, model, guide, numParticles, city):
    losses=[]
    errors=[]
    errorsFrac=[]
    for i in range(len(tests)):
        tests[i].alpha_paramShop = train.alpha_paramShop.cuda()
        tests[i].alpha_paramSchool = train.alpha_paramSchool.cuda()
        tests[i].alpha_paramReligion = train.alpha_paramReligion.cuda()
        tests[i].beta_paramShop = train.beta_paramShop.cuda()
        tests[i].beta_paramSchool = train.beta_paramSchool.cuda()
        tests[i].beta_paramReligion = train.beta_paramReligion.cuda()
        tests[i].multiVisitVarShParam = train.multiVisitVarShParam
        tests[i].multiVisitVarSchParam = train.multiVisitVarSchParam
        tests[i].multiVisitVarRelParam = train.multiVisitVarRelParam
        tests[i].gapParam = train.gapParam
        tests[i].isTrainedOnOneMonth=train.isTrainedOnOneMonth
        tests[i].globalError = np.zeros(numParticles, dtype=np.float32)
        tests[i].globalErrorFrac = np.zeros(numParticles, dtype=np.float32)
        loss = elbo.loss(model, guide, [tests[i]])
        logging.info("final loss test {} = {}".format(city,loss))
        logging.info("final error test {} = {}".format(city,tests[i].globalError))
        logging.info("final error frac test {} = {}".format(city,tests[i].globalErrorFrac))
        losses.append(loss)
        errors.append(tests[i].globalError[0])
        errorsFrac.append(tests[i].globalErrorFrac[0])

    # now = datetime.now()
    # dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    # with open('tests' + os.sep + 'validation_{}_{}_{}.csv'.format(dt_string, city, extraMessage), 'w', encoding='UTF8', newline='') as f:
    #     writer = csv.writer(f)
    #     for i in range(len(losses)):
    #         writer.writerow([losses[i]])
    return losses,errors,errorsFrac