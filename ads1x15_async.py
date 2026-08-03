# The MIT License (MIT)
#
# Copyright (c) 2016 Radomir Dopieralski (@deshipu),
#               2017 Robert Hammelrath (@robert-hh)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Standalone Async/Interrupt-driven driver for the ADS1115 ADC.
#
# This driver is completely independent of the standard synchronous driver
# (ads1x15.py). It implements only the necessary asynchronous conversion-ready
# reading method (aioread) using a hardware ALERT/RDY pin interrupt, leaving
# all synchronous and unused polling/comparator methods behind.
#
import asyncio
import machine

_REGISTER_CONVERT = const(0x00)
_REGISTER_CONFIG = const(0x01)
_REGISTER_LOWTHRESH = const(0x02)
_REGISTER_HITHRESH = const(0x03)

_OS_SINGLE = const(0x8000)  # Write: Set to start a single-conversion

_MUX_DIFF_0_1 = const(0x0000)  # Differential P = AIN0, N = AIN1
_MUX_DIFF_0_3 = const(0x1000)  # Differential P = AIN0, N = AIN3
_MUX_DIFF_1_3 = const(0x2000)  # Differential P = AIN1, N = AIN3
_MUX_DIFF_2_3 = const(0x3000)  # Differential P = AIN2, N = AIN3
_MUX_SINGLE_0 = const(0x4000)  # Single-ended AIN0
_MUX_SINGLE_1 = const(0x5000)  # Single-ended AIN1
_MUX_SINGLE_2 = const(0x6000)  # Single-ended AIN2
_MUX_SINGLE_3 = const(0x7000)  # Single-ended AIN3

_PGA_6_144V = const(0x0000)  # +/-6.144V range = Gain 2/3
_PGA_4_096V = const(0x0200)  # +/-4.096V range = Gain 1
_PGA_2_048V = const(0x0400)  # +/-2.048V range = Gain 2
_PGA_1_024V = const(0x0600)  # +/-1.024V range = Gain 4
_PGA_0_512V = const(0x0800)  # +/-0.512V range = Gain 8
_PGA_0_256V = const(0x0A00)  # +/-0.256V range = Gain 16

_MODE_SINGLE = const(0x0100)  # Power-down single-shot mode

_DR_128SPS = const(0x0000)   # 8 samples per second (ADS1115)
_DR_250SPS = const(0x0020)   # 16 samples per second
_DR_490SPS = const(0x0040)   # 32 samples per second
_DR_920SPS = const(0x0060)   # 64 samples per second
_DR_1600SPS = const(0x0080)  # 128 samples per second (default)
_DR_2400SPS = const(0x00A0)  # 250 samples per second
_DR_3300SPS = const(0x00C0)  # 475 samples per second
_DR_860SPS = const(0x00E0)   # 860 samples per second

_CMODE_TRAD = const(0x0000)   # Traditional comparator with hysteresis
_CPOL_ACTVLOW = const(0x0000) # ALERT/RDY pin is low when active
_CLAT_NONLAT = const(0x0000)  # Non-latching comparator
_CQUE_1CONV = const(0x0000)   # Assert ALERT/RDY after one conversion

_GAINS = (
    _PGA_6_144V,
    _PGA_4_096V,
    _PGA_2_048V,
    _PGA_1_024V,
    _PGA_0_512V,
    _PGA_0_256V
)

_GAINS_V = (
    6.144,
    4.096,
    2.048,
    1.024,
    0.512,
    0.256
)

_CHANNELS = {
    (0, None): _MUX_SINGLE_0,
    (1, None): _MUX_SINGLE_1,
    (2, None): _MUX_SINGLE_2,
    (3, None): _MUX_SINGLE_3,
    (0, 1): _MUX_DIFF_0_1,
    (0, 3): _MUX_DIFF_0_3,
    (1, 3): _MUX_DIFF_1_3,
    (2, 3): _MUX_DIFF_2_3,
}

_RATES = (
    _DR_128SPS,
    _DR_250SPS,
    _DR_490SPS,
    _DR_920SPS,
    _DR_1600SPS,
    _DR_2400SPS,
    _DR_3300SPS,
    _DR_860SPS
)


class ADS1115Async:
    """ADS1115 driver with interrupt-driven async read.

    The ALERT/RDY pin must be connected and passed as ready_pin. The pin is
    configured with a pull-up (ALERT/RDY is active-low) and a falling-edge IRQ
    that wakes the waiting coroutine via asyncio.ThreadSafeFlag.

    Usage::

        from machine import I2C, Pin
        from ads1x15_async import ADS1115Async
        import asyncio

        i2c = I2C(0, scl=Pin(12), sda=Pin(11), freq=400_000)
        adc = ADS1115Async(i2c, address=0x48, gain=1,
                           ready_pin=Pin(9, Pin.IN))

        async def main():
            value = await adc.aioread(rate=4, channel1=0)
            print("Voltage:", adc.raw_to_v(value))

        asyncio.run(main())
    """

    def __init__(self, i2c, address=0x48, gain=1, ready_pin=None):
        self.i2c = i2c
        self.address = address
        self.gain = gain
        self.temp2 = bytearray(2)
        self._lock = asyncio.Lock()

        if ready_pin is None:
            raise ValueError("ready_pin is required for ADS1115Async")
        self._ready_pin = ready_pin
        # ALERT/RDY is active-low; pull-up keeps the line stable between reads
        self._ready_pin.init(pull=machine.Pin.PULL_UP)
        # Thread-safe flag is the correct bridge between an IRQ and asyncio
        self._ready_flag = asyncio.ThreadSafeFlag()
        # Attach falling-edge IRQ (pin goes low when conversion is ready)
        self._ready_irq = self._ready_pin.irq(
            trigger=machine.Pin.IRQ_FALLING,
            handler=self._ready_callback,
        )
        # Enable conversion-ready mode on ALERT/RDY (datasheet §7.3.8).
        # Lo_thresh MSB = 0 and Hi_thresh MSB = 1 are the sentinel values.
        self._write_register(_REGISTER_LOWTHRESH, 0x0000)
        self._write_register(_REGISTER_HITHRESH, 0x8000)

    def _ready_callback(self, pin):
        """IRQ handler – must be minimal; only sets the flag."""
        self._ready_flag.set()

    def __del__(self):
        """Disable the IRQ when the object is reclaimed."""
        if hasattr(self, "_ready_irq") and self._ready_irq is not None:
            try:
                self._ready_irq.disable()
            except Exception:
                pass

    def _write_register(self, register, value):
        self.temp2[0] = value >> 8
        self.temp2[1] = value & 0xff
        self.i2c.writeto_mem(self.address, register, self.temp2)

    def _read_register(self, register):
        self.i2c.readfrom_mem_into(self.address, register, self.temp2)
        return (self.temp2[0] << 8) | self.temp2[1]

    def raw_to_v(self, raw):
        v_p_b = _GAINS_V[self.gain] / 32768
        return raw * v_p_b

    async def aioread(self, rate=4, channel1=0, channel2=None):
        """Start a single-shot conversion and wait for the ALERT/RDY interrupt.

        The lock ensures that concurrent coroutines reading different channels
        are serialised correctly.
        """
        async with self._lock:
            # Start single-shot conversion; CQUE_1CONV keeps ALERT/RDY enabled
            self._write_register(_REGISTER_CONFIG, (
                _CQUE_1CONV | _CLAT_NONLAT |
                _CPOL_ACTVLOW | _CMODE_TRAD | _RATES[rate] |
                _MODE_SINGLE | _OS_SINGLE | _GAINS[self.gain] |
                _CHANNELS[(channel1, channel2)]
            ))
            # Block this coroutine until the IRQ fires; other coroutines run freely
            await self._ready_flag.wait()

        res = self._read_register(_REGISTER_CONVERT)
        return res if res < 32768 else res - 65536
