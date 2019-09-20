<?php
/*
 * Copyright 2015 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

namespace Google\FlatBuffers;

abstract class Table
{
    /**
     * @var int $bb_pos
     */
    protected $bb_pos;
    /**
     * @var ByteBuffer $bb
     */
    protected $bb;

    public function __construct()
    {
    }

    public function setByteBufferPos($pos)
    {
        $this->bb_pos = $pos;
    }

    public function setByteBuffer($bb)
    {
        $this->bb = $bb;
    }

    /**
     * returns actual vtable offset
     *
     * @param $vtable_offset
     * @return int offset > 0 means exist value. 0 means not exist
     */
    protected function __offset($vtable_offset)
    {
        $vtable = $this->bb_pos - $this->bb->getInt($this->bb_pos);
        return $vtable_offset < $this->bb->getShort($vtable) ? $this->bb->getShort($vtable + $vtable_offset) : 0;
    }

    /**
     * @param $offset
     * @return mixed
     */
    protected function __indirect($offset)
    {
        return $offset + $this->bb->getInt($offset);
    }

    /**
     * fetch utf8 encoded string.
     *
     * @param $offset
     * @return string
     */
    protected function __string($offset)
    {
        $offset += $this->bb->getInt($offset);
        $len = $this->bb->getInt($offset);
        $startPos = $offset + Constants::SIZEOF_INT;
        return substr($this->bb->_buffer, $startPos, $len);
    }

    /**
     * @param $offset
     * @return int
     */
    protected function __vector_len($offset)
    {
        $offset += $this->bb_pos;
        $offset += $this->bb->getInt($offset);
        return $this->bb->getInt($offset);
    }

    /**
     * @param $offset
     * @return int
     */
    protected function __vector($offset)
    {
        $offset += $this->bb_pos;
        // data starts after the length
        return $offset + $this->bb->getInt($offset) + Constants::SIZEOF_INT;
    }

    protected function __vector_as_bytes($vector_offset, $elem_size=1)
    {
        $o = $this->__offset($vector_offset);
        if ($o == 0) {
            return null;
        }

        return substr($this->bb->_buffer, $this->__vector($o), $this->__vector_len($o) * $elem_size);
    }

    /**
     * @param Table $table
     * @param int $offset
     * @return Table
     */
    protected function __union($table, $offset)
    {
        $offset += $this->bb_pos;
        $table->setByteBufferPos($offset + $this->bb->getInt($offset));
        $table->setByteBuffer($this->bb);
        return $table;
    }

    /**
     * @param ByteBuffer $bb
     * @param string $ident
     * @return bool
     * @throws \ArgumentException
     */
    protected static function __has_identifier($bb, $ident)
    {
        if (strlen($ident) != Constants::FILE_IDENTIFIER_LENGTH) {
            throw new \ArgumentException("FlatBuffers: file identifier must be length "  . Constants::FILE_IDENTIFIER_LENGTH);
        }

        for ($i = 0; $i < 4; $i++) {
            if ($ident[$i] != $bb->get($bb->getPosition() + Constants::SIZEOF_INT + $i)) {
                return false;
            }
        }

        return true;
    }
}
