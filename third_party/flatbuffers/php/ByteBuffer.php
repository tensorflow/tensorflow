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

class ByteBuffer
{
    /**
     * @var string $_buffer;
     */
    public $_buffer;

    /**
     * @var int $_pos;
     */
    private $_pos;

    /**
     * @var bool $_is_little_endian
     */
    private static $_is_little_endian = null;

    public static function wrap($bytes)
    {
        $bb = new ByteBuffer(0);
        $bb->_buffer = $bytes;

        return $bb;
    }

    /**
     * @param $size
     */
    public function __construct($size)
    {
        $this->_buffer = str_repeat("\0", $size);
    }

    /**
     * @return int
     */
    public function capacity()
    {
        return strlen($this->_buffer);
    }

    /**
     * @return int
     */
    public function getPosition()
    {
        return $this->_pos;
    }

    /**
     * @param $pos
     */
    public function setPosition($pos)
    {
        $this->_pos = $pos;
    }

    /**
     *
     */
    public function reset()
    {
        $this->_pos = 0;
    }

    /**
     * @return int
     */
    public function length()
    {
        return strlen($this->_buffer);
    }

    /**
     * @return string
     */
    public function data()
    {
        return substr($this->_buffer, $this->_pos);
    }

    /**
     * @return bool
     */
    public static function isLittleEndian()
    {
        if (ByteBuffer::$_is_little_endian === null) {
            ByteBuffer::$_is_little_endian = unpack('S', "\x01\x00")[1] === 1;
        }

        return ByteBuffer::$_is_little_endian;
    }

    /**
     * write little endian value to the buffer.
     *
     * @param $offset
     * @param $count byte length
     * @param $data actual values
     */
    public function writeLittleEndian($offset, $count, $data)
    {
        if (ByteBuffer::isLittleEndian()) {
            for ($i = 0; $i < $count; $i++) {
                $this->_buffer[$offset + $i] = chr($data >> $i * 8);
            }
        } else {
            for ($i = 0; $i < $count; $i++) {
                $this->_buffer[$offset + $count - 1 - $i] = chr($data >> $i * 8);
            }
        }
    }

    /**
     * read little endian value from the buffer
     *
     * @param $offset
     * @param $count acutal size
     * @return int
     */
    public function readLittleEndian($offset, $count, $force_bigendian = false)
    {
        $this->assertOffsetAndLength($offset, $count);
        $r = 0;

        if (ByteBuffer::isLittleEndian() && $force_bigendian == false) {
            for ($i = 0; $i < $count; $i++) {
                $r |= ord($this->_buffer[$offset + $i]) << $i * 8;
            }
        } else {
            for ($i = 0; $i < $count; $i++) {
                $r |= ord($this->_buffer[$offset + $count -1 - $i]) << $i * 8;
            }
        }

        return $r;
    }

    /**
     * @param $offset
     * @param $length
     */
    public function assertOffsetAndLength($offset, $length)
    {
        if ($offset < 0 ||
            $offset >= strlen($this->_buffer) ||
            $offset + $length > strlen($this->_buffer)) {
            throw new \OutOfRangeException(sprintf("offset: %d, length: %d, buffer; %d", $offset, $length, strlen($this->_buffer)));
        }
    }

    /**
     * @param $offset
     * @param $value
     * @return mixed
     */
    public function putSbyte($offset, $value)
    {
        self::validateValue(-128, 127, $value, "sbyte");

        $length = strlen($value);
        $this->assertOffsetAndLength($offset, $length);
        return $this->_buffer[$offset] = $value;
    }

    /**
     * @param $offset
     * @param $value
     * @return mixed
     */
    public function putByte($offset, $value)
    {
        self::validateValue(0, 255, $value, "byte");

        $length = strlen($value);
        $this->assertOffsetAndLength($offset, $length);
        return $this->_buffer[$offset] = $value;
    }

    /**
     * @param $offset
     * @param $value
     */
    public function put($offset, $value)
    {
        $length = strlen($value);
        $this->assertOffsetAndLength($offset, $length);
        for ($i = 0; $i < $length; $i++) {
            $this->_buffer[$offset + $i] = $value[$i];
        }
    }

    /**
     * @param $offset
     * @param $value
     */
    public function putShort($offset, $value)
    {
        self::validateValue(-32768, 32767, $value, "short");

        $this->assertOffsetAndLength($offset, 2);
        $this->writeLittleEndian($offset, 2, $value);
    }

    /**
     * @param $offset
     * @param $value
     */
    public function putUshort($offset, $value)
    {
        self::validateValue(0, 65535, $value, "short");

        $this->assertOffsetAndLength($offset, 2);
        $this->writeLittleEndian($offset, 2, $value);
    }

    /**
     * @param $offset
     * @param $value
     */
    public function putInt($offset, $value)
    {
        // 2147483647 = (1 << 31) -1 = Maximum signed 32-bit int
        // -2147483648 = -1 << 31 = Minimum signed 32-bit int
        self::validateValue(-2147483648, 2147483647, $value, "int");

        $this->assertOffsetAndLength($offset, 4);
        $this->writeLittleEndian($offset, 4, $value);
    }

    /**
     * @param $offset
     * @param $value
     */
    public function putUint($offset, $value)
    {
        // NOTE: We can't put big integer value. this is PHP limitation.
        // 4294967295 = (1 << 32) -1 = Maximum unsigned 32-bin int
        self::validateValue(0, 4294967295, $value, "uint",  " php has big numbers limitation. check your PHP_INT_MAX");

        $this->assertOffsetAndLength($offset, 4);
        $this->writeLittleEndian($offset, 4, $value);
    }

    /**
     * @param $offset
     * @param $value
     */
    public function putLong($offset, $value)
    {
        // NOTE: We can't put big integer value. this is PHP limitation.
        self::validateValue(~PHP_INT_MAX, PHP_INT_MAX, $value, "long",  " php has big numbers limitation. check your PHP_INT_MAX");

        $this->assertOffsetAndLength($offset, 8);
        $this->writeLittleEndian($offset, 8, $value);
    }

    /**
     * @param $offset
     * @param $value
     */
    public function putUlong($offset, $value)
    {
        // NOTE: We can't put big integer value. this is PHP limitation.
        self::validateValue(0, PHP_INT_MAX, $value, "long", " php has big numbers limitation. check your PHP_INT_MAX");

        $this->assertOffsetAndLength($offset, 8);
        $this->writeLittleEndian($offset, 8, $value);
    }

    /**
     * @param $offset
     * @param $value
     */
    public function putFloat($offset, $value)
    {
        $this->assertOffsetAndLength($offset, 4);

        $floathelper = pack("f", $value);
        $v = unpack("V", $floathelper);
        $this->writeLittleEndian($offset, 4, $v[1]);
    }

    /**
     * @param $offset
     * @param $value
     */
    public function putDouble($offset, $value)
    {
        $this->assertOffsetAndLength($offset, 8);

        $floathelper = pack("d", $value);
        $v = unpack("V*", $floathelper);

        $this->writeLittleEndian($offset, 4, $v[1]);
        $this->writeLittleEndian($offset + 4, 4, $v[2]);
    }

    /**
     * @param $index
     * @return mixed
     */
    public function getByte($index)
    {
        return ord($this->_buffer[$index]);
    }

    /**
     * @param $index
     * @return mixed
     */
    public function getSbyte($index)
    {
        $v = unpack("c", $this->_buffer[$index]);
        return $v[1];
    }

    /**
     * @param $buffer
     */
    public function getX(&$buffer)
    {
        for ($i = $this->_pos, $j = 0; $j < strlen($buffer); $i++, $j++) {
            $buffer[$j] = $this->_buffer[$i];
        }
    }

    /**
     * @param $index
     * @return mixed
     */
    public function get($index)
    {
        $this->assertOffsetAndLength($index, 1);
        return $this->_buffer[$index];
    }


    /**
     * @param $index
     * @return mixed
     */
    public function getBool($index)
    {
        return (bool)ord($this->_buffer[$index]);
    }

    /**
     * @param $index
     * @return int
     */
    public function getShort($index)
    {
        $result = $this->readLittleEndian($index, 2);

        $sign = $index + (ByteBuffer::isLittleEndian() ? 1 : 0);
        $issigned = isset($this->_buffer[$sign]) && ord($this->_buffer[$sign]) & 0x80;

        // 65536 = 1 << 16 = Maximum unsigned 16-bit int
        return $issigned ? $result - 65536 : $result;
    }

    /**
     * @param $index
     * @return int
     */
    public function getUShort($index)
    {
        return $this->readLittleEndian($index, 2);
    }

    /**
     * @param $index
     * @return int
     */
    public function getInt($index)
    {
        $result = $this->readLittleEndian($index, 4);

        $sign = $index + (ByteBuffer::isLittleEndian() ? 3 : 0);
        $issigned = isset($this->_buffer[$sign]) && ord($this->_buffer[$sign]) & 0x80;

        if (PHP_INT_SIZE > 4) {
            // 4294967296 = 1 << 32 = Maximum unsigned 32-bit int
            return $issigned ? $result - 4294967296 : $result;
        } else {
            // 32bit / Windows treated number as signed integer.
            return $result;
        }
    }

    /**
     * @param $index
     * @return int
     */
    public function getUint($index)
    {
        return $this->readLittleEndian($index, 4);
    }

    /**
     * @param $index
     * @return int
     */
    public function getLong($index)
    {
        return $this->readLittleEndian($index, 8);
    }

    /**
     * @param $index
     * @return int
     */
    public function getUlong($index)
    {
        return $this->readLittleEndian($index, 8);
    }

    /**
     * @param $index
     * @return mixed
     */
    public function getFloat($index)
    {
        $i = $this->readLittleEndian($index, 4);

        return self::convertHelper(self::__FLOAT, $i);
    }

    /**
     * @param $index
     * @return float
     */
    public function getDouble($index)
    {
        $i = $this->readLittleEndian($index, 4);
        $i2 = $this->readLittleEndian($index + 4, 4);

        return self::convertHelper(self::__DOUBLE, $i, $i2);
    }

    const __SHORT = 1;
    const __INT = 2;
    const __LONG = 3;
    const __FLOAT = 4;
    const __DOUBLE = 5;
    private static function convertHelper($type, $value, $value2 = null) {
        // readLittleEndian construct unsigned integer value from bytes. we have to encode this value to
        // correct bytes, and decode as expected types with `unpack` function.
        // then it returns correct type value.
        // see also: http://php.net/manual/en/function.pack.php

        switch ($type) {
            case self::__FLOAT:
                $inthelper = pack("V", $value);
                $v = unpack("f", $inthelper);
                return $v[1];
                break;
            case self::__DOUBLE:
                $inthelper = pack("VV", $value, $value2);
                $v = unpack("d", $inthelper);
                return $v[1];
                break;
            default:
                throw new \Exception(sprintf("unexpected type %d specified", $type));
        }
    }

    private static function validateValue($min, $max, $value, $type, $additional_notes = "") {
        if(!($min <= $value && $value <= $max)) {
            throw new \InvalidArgumentException(sprintf("bad number %s for type %s.%s", $value, $type, $additional_notes));
        }
    }
}
