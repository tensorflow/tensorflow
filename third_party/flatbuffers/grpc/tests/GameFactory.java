import java.nio.ByteBuffer;
import MyGame.Example.Monster;
import MyGame.Example.Stat;
import com.google.flatbuffers.FlatBufferBuilder;

class GameFactory {
  public static Monster createMonster(String monsterName, short nestedMonsterHp, short nestedMonsterMana) {
    FlatBufferBuilder builder = new FlatBufferBuilder();

    int name_offset = builder.createString(monsterName);
    Monster.startMonster(builder);
    Monster.addName(builder, name_offset);
    Monster.addHp(builder, nestedMonsterHp);
    Monster.addMana(builder, nestedMonsterMana);
    int monster_offset = Monster.endMonster(builder);
    Monster.finishMonsterBuffer(builder, monster_offset);

    ByteBuffer buffer = builder.dataBuffer();
    Monster monster = Monster.getRootAsMonster(buffer);
    return monster;
  }

  public static Monster createMonsterFromStat(Stat stat, int seqNo) {
    FlatBufferBuilder builder = new FlatBufferBuilder();
    int name_offset = builder.createString(stat.id() + " No." + seqNo);
    Monster.startMonster(builder);
    Monster.addName(builder, name_offset);
    int monster_offset = Monster.endMonster(builder);
    Monster.finishMonsterBuffer(builder, monster_offset);
    Monster monster = Monster.getRootAsMonster(builder.dataBuffer());
    return monster;
  }

  public static Stat createStat(String greeting, long val, int count) { 
    FlatBufferBuilder builder = new FlatBufferBuilder();
    int statOffset = Stat.createStat(builder, builder.createString(greeting), val, count);
    builder.finish(statOffset);
    Stat stat = Stat.getRootAsStat(builder.dataBuffer());
    return stat;
  }

}
