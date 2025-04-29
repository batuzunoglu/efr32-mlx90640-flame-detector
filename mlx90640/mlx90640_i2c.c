//
#include <stdlib.h> // Required for malloc() and free()

#include "mlx90640_i2c.h"
#include "stdio.h"
#define MLX90640_DEFAULT_I2C_ADDR   0x33

static sl_i2cspm_t *i2cspm;
static uint8_t i2c_addr = MLX90640_DEFAULT_I2C_ADDR;

/***************************************************************************//**
 * Assigns an I2CSPM instance for the driver to use
 ******************************************************************************/
sl_status_t mlx90640_I2C_Init(sl_i2cspm_t *i2cspm_instance)
{
  i2cspm = i2cspm_instance;
  i2c_addr = MLX90640_DEFAULT_I2C_ADDR;
  return SL_STATUS_OK;
}

/***************************************************************************//**
 * Issues an I2C general reset
 ******************************************************************************/
sl_status_t mlx90640_I2CGeneralReset(void)
{
  I2C_TransferSeq_TypeDef seq;
  I2C_TransferReturn_TypeDef ret;

  uint8_t cmd[2] = { 0x00, 0x06 };

  seq.addr = i2c_addr;
  seq.flags = I2C_FLAG_WRITE;
  seq.buf[0].len = 2;
  seq.buf[0].data = cmd;

  ret = I2CSPM_Transfer(i2cspm, &seq);

  if (ret != i2cTransferDone) {
    return -1;
  }
  return SL_STATUS_OK;
}

/***************************************************************************//**
 * Initiates an I2C read of the device
 ******************************************************************************/
sl_status_t mlx90640_I2CRead(uint16_t startAddress, uint16_t nMemAddressRead, uint16_t *data)
{
    uint8_t *i2cData = (uint8_t*)malloc(2 * nMemAddressRead); // Heap allocation
    if (i2cData == NULL) {
        printf("\r\n Handle allocation failure..!\n");
        return SL_STATUS_FAIL; // Handle allocation failure
    }

    uint16_t counter = 0;
    uint16_t i = 0;
    uint16_t *p = data;
    I2C_TransferSeq_TypeDef seq;
    I2C_TransferReturn_TypeDef ret;

    uint8_t cmd[2] = { startAddress >> 8, startAddress & 0x00FF };

    seq.addr = i2c_addr << 1;
    seq.flags = I2C_FLAG_WRITE_READ;
    seq.buf[0].len = 2;
    seq.buf[0].data = cmd;
    seq.buf[1].len = 2 * nMemAddressRead;
    seq.buf[1].data = i2cData;
    ret = I2CSPM_Transfer(i2cspm, &seq);

    if (ret != i2cTransferDone) {
        free(i2cData);
        return SL_STATUS_FAIL;
    }

    for (counter = 0; counter < nMemAddressRead; counter++) {
        i = counter << 1;
        *p++ = (uint16_t)i2cData[i] * 256 + (uint16_t)i2cData[i + 1];
    }

    free(i2cData); // Free memory after use
    return SL_STATUS_OK;
}

/***************************************************************************//**
 * Sets I2C base frequency to a given setting
 ******************************************************************************/
void mlx90640_I2CFreqSet(int freq)
{
  I2C_BusFreqSet(i2cspm, 0, freq, i2cClockHLRStandard); //Todo or 1000*freq?
}

/***************************************************************************//**
 * Initiates an I2C write to the device
 ******************************************************************************/
sl_status_t mlx90640_I2CWrite(uint16_t writeAddress, uint16_t data)
{
  uint8_t cmd[4] = { 0, 0, 0, 0 };
  static uint16_t dataCheck;

  I2C_TransferSeq_TypeDef seq;
  I2C_TransferReturn_TypeDef i2c_ret;
  int ret;

  cmd[0] = writeAddress >> 8;
  cmd[1] = writeAddress & 0x00FF;
  cmd[2] = data >> 8;
  cmd[3] = data & 0x00FF;

  seq.addr = i2c_addr << 1;
  seq.flags = I2C_FLAG_WRITE;
  seq.buf[0].len = 4;
  seq.buf[0].data = cmd;

  i2c_ret = I2CSPM_Transfer(i2cspm, &seq);

  if (i2c_ret != i2cTransferDone) {
    return SL_STATUS_FAIL;
  }

  ret = mlx90640_I2CRead(writeAddress, 1, &dataCheck);

  if (ret != 0) {
    return SL_STATUS_FAIL;
  }

  if (dataCheck != data) {
    return SL_STATUS_FAIL;   //Todo data error?
  }

  return SL_STATUS_OK;
}
