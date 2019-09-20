<?php

require join(DIRECTORY_SEPARATOR, array(dirname(dirname(__FILE__)), "php", "Constants.php"));
require join(DIRECTORY_SEPARATOR, array(dirname(dirname(__FILE__)), "php", "ByteBuffer.php"));
require join(DIRECTORY_SEPARATOR, array(dirname(dirname(__FILE__)), "php", "FlatbufferBuilder.php"));
require join(DIRECTORY_SEPARATOR, array(dirname(dirname(__FILE__)), "php", "Table.php"));
require join(DIRECTORY_SEPARATOR, array(dirname(dirname(__FILE__)), "php", "Struct.php"));

require join(DIRECTORY_SEPARATOR, array(dirname(__FILE__), "php", 'Attacker.php'));
require join(DIRECTORY_SEPARATOR, array(dirname(__FILE__), "php", 'BookReader.php'));
require join(DIRECTORY_SEPARATOR, array(dirname(__FILE__), "php", 'Character.php'));
require join(DIRECTORY_SEPARATOR, array(dirname(__FILE__), "php", 'Movie.php'));

class Assert {
    public function ok($result, $message = "") {
        if (!$result){
            throw new Exception(!empty($message) ? $message : "{$result} is not true.");
        }
    }

    public function Equal($result, $expected, $message = "") {
        if ($result != $expected) {
            throw new Exception(!empty($message) ? $message : "given the result {$result} is not equals as {$expected}");
        }
    }


    public function strictEqual($result, $expected, $message = "") {
        if ($result !== $expected) {
            throw new Exception(!empty($message) ? $message : "given the result {$result} is not strict equals as {$expected}");
        }
    }

    public function Throws($class, Callable $callback) {
        try {
            $callback();

            throw new \Exception("passed statement don't throw an exception.");
        } catch (\Exception $e) {
            if (get_class($e) != get_class($class)) {
                throw new Exception("passed statement doesn't throw " . get_class($class) . ". throwws " . get_class($e));
            }
        }
    }
}

function main()
{
    $assert = new Assert();

    $fbb = new Google\FlatBuffers\FlatBufferBuilder(1);

    $charTypes = [
        Character::Belle,
        Character::MuLan,
        Character::BookFan,
    ];

    Attacker::startAttacker($fbb);
    Attacker::addSwordAttackDamage($fbb, 5);
    $attackerOffset = Attacker::endAttacker($fbb);

    $charTypesOffset = Movie::createCharactersTypeVector($fbb, $charTypes);
    $charsOffset = Movie::createCharactersVector(
        $fbb,
        [
            BookReader::createBookReader($fbb, 7),
            $attackerOffset,
            BookReader::createBookReader($fbb, 2),
        ]
    );

    Movie::startMovie($fbb);
    Movie::addCharactersType($fbb, $charTypesOffset);
    Movie::addCharacters($fbb, $charsOffset);
    Movie::finishMovieBuffer($fbb, Movie::endMovie($fbb));

    $buf = Google\FlatBuffers\ByteBuffer::wrap($fbb->dataBuffer()->data());

    $movie = Movie::getRootAsMovie($buf);

    $assert->strictEqual($movie->getCharactersTypeLength(), count($charTypes));
    $assert->strictEqual($movie->getCharactersLength(), $movie->getCharactersTypeLength());

    for ($i = 0; $i < count($charTypes); ++$i) {
        $assert->strictEqual($movie->getCharactersType($i), $charTypes[$i]);
    }

    $bookReader7 = $movie->getCharacters(0, new BookReader());
    $assert->strictEqual($bookReader7->getBooksRead(), 7);

    $attacker = $movie->getCharacters(1, new Attacker());
    $assert->strictEqual($attacker->getSwordAttackDamage(), 5);

    $bookReader2 = $movie->getCharacters(2, new BookReader());
    $assert->strictEqual($bookReader2->getBooksRead(), 2);
}

try {
    main();
    exit(0);
} catch(Exception $e) {
    printf("Fatal error: Uncaught exception '%s' with message '%s. in %s:%d\n", get_class($e), $e->getMessage(), $e->getFile(), $e->getLine());
    printf("Stack trace:\n");
    echo $e->getTraceAsString() . PHP_EOL;
    printf("  thrown in in %s:%d\n", $e->getFile(), $e->getLine());

    die(-1);
}
