var assert = require('assert');

var flatbuffers = require('../js/flatbuffers').flatbuffers;
var Test = require(process.argv[2]);

function main() {
  var fbb = new flatbuffers.Builder();

  var charTypes = [
    Test.Character.Belle,
    Test.Character.MuLan,
    Test.Character.BookFan,
  ];

  Test.Attacker.startAttacker(fbb);
  Test.Attacker.addSwordAttackDamage(fbb, 5);
  var attackerOffset = Test.Attacker.endAttacker(fbb);

  var charTypesOffset = Test.Movie.createCharactersTypeVector(fbb, charTypes);
  var charsOffset = Test.Movie.createCharactersVector(
    fbb,
    [
      Test.BookReader.createBookReader(fbb, 7),
      attackerOffset,
      Test.BookReader.createBookReader(fbb, 2),
    ]
  );

  Test.Movie.startMovie(fbb);
  Test.Movie.addCharactersType(fbb, charTypesOffset);
  Test.Movie.addCharacters(fbb, charsOffset);
  Test.Movie.finishMovieBuffer(fbb, Test.Movie.endMovie(fbb));

  var buf = new flatbuffers.ByteBuffer(fbb.asUint8Array());

  var movie = Test.Movie.getRootAsMovie(buf);

  assert.strictEqual(movie.charactersTypeLength(), charTypes.length);
  assert.strictEqual(movie.charactersLength(), movie.charactersTypeLength());

  for (var i = 0; i < charTypes.length; ++i) {
    assert.strictEqual(movie.charactersType(i), charTypes[i]);
  }

  var bookReader7 = movie.characters(0, new Test.BookReader());
  assert.strictEqual(bookReader7.booksRead(), 7);

  var attacker = movie.characters(1, new Test.Attacker());
  assert.strictEqual(attacker.swordAttackDamage(), 5);

  var bookReader2 = movie.characters(2, new Test.BookReader());
  assert.strictEqual(bookReader2.booksRead(), 2);

  console.log('FlatBuffers union vector test: completed successfully');
}

main();
