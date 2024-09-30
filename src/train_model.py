from keras.callbacks import ModelCheckpoint ,EarlyStopping

#early stopping
es = EarlyStopping(monitor= 'val_accuracy', min_delta= 0.01 ,patience = 3, verbose= 1)


#model check point
mc = ModelCheckpoint(filepath="best_model.h5",
                     monitor= 'val_accuracy',
                     min_delta= 0.01 ,
                     patience = 3,
                     verbose= 1,
                     save_best_only= True)



cb=[es,mc]


his = model.fit_generator(train ,
                          steps_per_epoch =16,
                          epochs= 50 ,
                          verbose = 1,
                          callbacks=cb ,
                          validation_data= val ,
                          validation_steps= 16 )

h = his.history
h.keys()

plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'] , c="red")
plt.title("acc vs v-acc")
plt.show()
