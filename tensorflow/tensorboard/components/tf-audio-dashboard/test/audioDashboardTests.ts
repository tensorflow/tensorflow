declare function stub(el: string, obj: any): void;

    describe('audio dashboard tests', function() {
      var audioDash;
      var reloadCount = 0;
      beforeEach(function() {
        audioDash = fixture('testElementFixture');
        var router = TF.Backend.router('data', true);
        var backend = new TF.Backend.Backend(router);
        audioDash.backend = backend;
        stub('tf-audio-loader', {
          reload: function() { reloadCount++; },
        });
      });

      it('calling reload on dashboard reloads the audio-loaders',
         function(done) {
           audioDash.backendReload().then(() => {
             reloadCount = 0;
             var loaders = [].slice.call(
                 audioDash.getElementsByTagName('tf-audio-loader'));
             audioDash.frontendReload();
             setTimeout(function() {
               assert.isAbove(reloadCount, 3);
               done();
             });
           });
         });
    });
