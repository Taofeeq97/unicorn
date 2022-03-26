def fake_predict(user_age):
    if int(user_age)>10:
        prediction='Survived'
    else:
        prediction='Supersurvived'
    return prediction