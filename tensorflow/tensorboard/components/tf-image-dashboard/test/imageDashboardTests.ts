declare function stub(el: string, obj: any): void;

    describe('image dashboard tests', function() {
      var imageDash;
      var reloadCount = 0;
      beforeEach(function() {
        imageDash = fixture('testElementFixture');
        var router = TF.Backend.router('data', true);
        var backend = new TF.Backend.Backend(router);
        imageDash.backend = backend;
        stub('tf-image-loader', {
          reload: function() { reloadCount++; },
        });
      });

      it('calling reload on dashboard reloads the image-loaders',
         function(done) {
           imageDash.backendReload().then(() => {
             reloadCount = 0;
             var loaders = [].slice.call(
                 imageDash.getElementsByTagName('tf-image-loader'));
             imageDash.frontendReload();
             setTimeout(function() {
               assert.isAbove(reloadCount, 3);
               done();
             });
           });
         });
    });
